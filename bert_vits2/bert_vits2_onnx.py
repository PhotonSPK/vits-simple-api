"""
Bert-VITS2 ONNX 推理封装
======================
使用 ONNX Runtime + OpenVINO EP 进行推理，支持 Intel GPU 硬件加速。
无可用 GPU 时自动回落 CPU。

使用前需先运行 onnx_export.py 导出模型：
    python -m bert_vits2.onnx_export synthesize --model-path ... --config-path ... --output-dir ...
    python -m bert_vits2.onnx_export bert --model-name ... --model-path ... --output-dir ...
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from bert_vits2 import commons
from bert_vits2.text import cleaned_text_to_sequence
from bert_vits2.text.cleaner import clean_text
from bert_vits2.utils import process_legacy_versions
from utils import get_hparams_from_file

logger = logging.getLogger(__name__)


class BertVITS2ONNX:
    """
    Bert-VITS2 ONNX 推理类。

    与原有 Bert_VITS2 保持相同的 infer() 接口，底层使用 ONNX Runtime。
    """

    def __init__(
        self,
        vits_path: str,
        config,
        model_handler,
        onnx_model_dir: str,
        device: str = "cpu",
    ):
        """
        Args:
            vits_path: PyTorch 模型权重路径（用于读取元数据，不加载到 torch）
            config: 模型配置（hps 对象或 config.json 路径）
            model_handler: ModelHandler 实例（提供 tokenizer、G2PW、情感模型等）
            onnx_model_dir: ONNX 模型所在目录
            device: 设备类型字符串（"cpu", "cuda" 等，仅用于日志）
        """
        self.vits_path = vits_path
        self.hps_ms = get_hparams_from_file(config) if isinstance(config, str) else config
        self.model_handler = model_handler
        self.onnx_model_dir = onnx_model_dir
        self.device = device

        # 模型元数据
        self.n_speakers = getattr(self.hps_ms.data, "n_speakers", 0)
        self.speakers = [
            item[0]
            for item in sorted(
                list(getattr(self.hps_ms.data, "spk2id", {"0": 0}).items()),
                key=lambda x: x[1],
            )
        ]
        self.sampling_rate = self.hps_ms.data.sampling_rate

        # 版本检测
        self.version = process_legacy_versions(self.hps_ms).lower().replace("-", "_")

        # 语言和 BERT 配置
        self.lang = getattr(self.hps_ms.data, "lang", ["zh"])
        self.bert_model_names = {}
        self.zh_bert_extra = False
        self.ja_bert_extra = False
        self.ja_bert_dim = 1024
        self._detect_version_features()

        # 符号表
        from bert_vits2.text import symbols
        self.symbols = symbols
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

        # 文本处理映射
        self.text_extra_str_map = {"zh": "", "ja": "", "en": ""}
        self.bert_extra_str_map = {"zh": "", "ja": "", "en": ""}

        # 加载 ONNX 会话
        self._load_synthesizer_onnx()
        self._load_bert_onnx()

        logger.info(f"BertVITS2ONNX 初始化完成，版本={self.version}，设备={device}")

    # ------------------------------------------------------------------
    # 版本特性检测
    # ------------------------------------------------------------------

    def _detect_version_features(self):
        """根据模型版本设置语言/BERT 特性标志"""
        from bert_vits2.text import num_tones

        self.num_tones = num_tones
        emotion_embedding = getattr(self.hps_ms.model, "emotion_embedding", None)

        if self.version in ["1.0", "1.0.0", "1.0.1"]:
            self.ja_bert_dim = 768
            self.text_extra_str_map.update({"zh": "_v100"})
        elif self.version in ["1.1.0_transition"]:
            self.ja_bert_dim = 768
            self.bert_model_names.update({"ja": "BERT_BASE_JAPANESE_V3"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v111"})
            self.bert_extra_str_map.update({"ja": "_v111"})
        elif self.version in ["1.1", "1.1.0", "1.1.1"]:
            self.ja_bert_dim = 768
            self.bert_model_names.update({"ja": "BERT_BASE_JAPANESE_V3"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v111"})
            self.bert_extra_str_map.update({"ja": "_v111"})
        elif self.version in ["2.0", "2.0.0", "2.0.1", "2.0.2"]:
            self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE"})
            self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
            self.text_extra_str_map.update({"zh": "_v100", "ja": "_v200", "en": "_v200"})
            self.bert_extra_str_map.update({"ja": "_v200", "en": "_v200"})
        elif self.version in ["2.1", "2.1.0"]:
            self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
        elif self.version in ["2.2", "2.2.0"]:
            self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
        elif self.version in ["2.3", "2.3.0"]:
            self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            self.bert_model_names.update({"en": "DEBERTA_V3_LARGE"})
            self.text_extra_str_map.update({"en": "_v230"})
        elif self.version in ["extra", "zh_clap"]:
            self.zh_bert_extra = True
            self.bert_model_names.update({"zh": "Erlangshen_MegatronBert_1.3B_Chinese"})
            self.bert_extra_str_map.update({"zh": "_extra"})
        elif self.version in ["extra_fix", "2.4", "2.4.0"]:
            self.zh_bert_extra = True
            self.bert_model_names.update({"zh": "Erlangshen_MegatronBert_1.3B_Chinese"})
            self.bert_extra_str_map.update({"zh": "_extra"})
            self.text_extra_str_map.update({"zh": "_v240"})
        elif self.version == "ja_extra":
            self.ja_bert_extra = True
            self.bert_model_names.update({"ja": "DEBERTA_V2_LARGE_JAPANESE_CHAR_WWM"})
            self.bert_extra_str_map.update({"ja": "_extra"})
            self.text_extra_str_map.update({"ja": "_extra"})

        if "zh" in self.lang and "zh" not in self.bert_model_names:
            self.bert_model_names.update({"zh": "CHINESE_ROBERTA_WWM_EXT_LARGE"})

    # ------------------------------------------------------------------
    # ONNX 会话加载
    # ------------------------------------------------------------------

    def _get_ort_session(self, model_file: str):
        """创建 ONNX Runtime 会话，优先 OpenVINO，回落 CPU"""
        import onnxruntime as ort

        model_path = os.path.join(self.onnx_model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # OpenVINOExecutionProvider 部分算子不支持的回落 CPU
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            actual_providers = session.get_providers()
            logger.info(f"加载 ONNX: {model_file} → providers={actual_providers}")
            return session
        except Exception as e:
            logger.warning(f"OpenVINO 加载失败 ({e})，使用 CPU 回落")
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
            logger.info(f"加载 ONNX (CPU fallback): {model_file}")
            return session

    def _load_synthesizer_onnx(self):
        """加载 SynthesizerTrn ONNX 模型"""
        meta_path = os.path.join(self.onnx_model_dir, "synthesizer_meta.json")

        # 尝试加载元数据
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.syn_meta = json.load(f)
        else:
            self.syn_meta = {}

        self.syn_session = self._get_ort_session("synthesizer.onnx")
        self.syn_inputs = [inp.name for inp in self.syn_session.get_inputs()]
        logger.info(f"SynthesizerTrn 输入: {self.syn_inputs}")

    def _load_bert_onnx(self):
        """加载所有需要的 BERT ONNX 模型"""
        self.bert_sessions: Dict[str, "ort.InferenceSession"] = {}

        for lang, bert_name in self.bert_model_names.items():
            model_file = f"bert_{bert_name}.onnx"
            model_path = os.path.join(self.onnx_model_dir, model_file)

            if os.path.exists(model_path):
                self.bert_sessions[lang] = self._get_ort_session(model_file)
                logger.info(f"BERT ONNX 已加载: {lang} → {bert_name}")
            else:
                logger.warning(
                    f"BERT ONNX 模型不存在: {model_path}，将使用 PyTorch BERT 回落"
                )
                self.bert_sessions[lang] = None

    # ------------------------------------------------------------------
    # BERT ONNX 推理
    # ------------------------------------------------------------------

    def _onnx_bert_forward(
        self, lang: str, input_ids: np.ndarray, attention_mask: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """运行 BERT ONNX 推理，返回 last_hidden_state"""
        session = self.bert_sessions.get(lang)
        bert_name = self.bert_model_names.get(lang, "")

        if session is None:
            # 回落：使用 PyTorch BERT 模型
            return self._pytorch_bert_forward(lang, input_ids, attention_mask, token_type_ids)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and "token_type_ids" in [i.name for i in session.get_inputs()]:
            feed["token_type_ids"] = token_type_ids

        outputs = session.run(None, feed)
        return outputs[0]  # last_hidden_state: [1, seq_len, hidden_dim]

    def _pytorch_bert_forward(
        self, lang: str, input_ids: np.ndarray, attention_mask: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """PyTorch BERT 回落推理"""
        bert_name = self.bert_model_names.get(lang, "")
        tokenizer, model = self.model_handler.get_bert_model(bert_name)

        input_ids_t = torch.from_numpy(input_ids).to(self.model_handler.device)
        attention_mask_t = torch.from_numpy(attention_mask).to(self.model_handler.device)

        with torch.no_grad():
            if token_type_ids is not None:
                token_type_ids_t = torch.from_numpy(token_type_ids).to(self.model_handler.device)
                output = model(input_ids_t, attention_mask_t, token_type_ids_t)
            else:
                output = model(input_ids_t, attention_mask_t)

        if hasattr(output, "last_hidden_state"):
            result = output.last_hidden_state.cpu().numpy()
        else:
            result = output[0].cpu().numpy()
        return result

    # ------------------------------------------------------------------
    # 文本处理（复用 Bert_VITS2 的 get_text 逻辑）
    # ------------------------------------------------------------------

    def get_text(
        self, text: str, language_str: str, style_text=None, style_weight=0.7
    ) -> Tuple[torch.Tensor, ...]:
        """
        文本 → BERT 特征 + phonemes。

        返回与 Bert_VITS2.get_text() 相同的格式：
        (zh_bert, ja_bert, en_bert, phone, tone, language)
        """
        clean_text_lang_str = language_str + self.text_extra_str_map.get(language_str, "")
        bert_feature_lang_str = language_str + self.bert_extra_str_map.get(language_str, "")

        tokenizer, _ = self.model_handler.get_bert_model(self.bert_model_names[language_str])

        norm_text, phone, tone, word2ph = clean_text(
            text, clean_text_lang_str, tokenizer, self.pinyin_g2pw
        )

        phone_list, tone_list, language_list = cleaned_text_to_sequence(
            phone, tone, language_str, self._symbol_to_id
        )

        if self.hps_ms.data.add_blank:
            phone_list = commons.intersperse(phone_list, 0)
            tone_list = commons.intersperse(tone_list, 0)
            language_list = commons.intersperse(language_list, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        # BERT 特征获取（使用 ONNX 或回落 PyTorch）
        bert_feature = self._get_bert_feature_onnx(
            norm_text, word2ph, bert_feature_lang_str,
            self.bert_model_names[language_str], style_text, style_weight
        )

        assert bert_feature.shape[-1] == len(phone_list), (
            f"Bert seq len {bert_feature.shape[-1]} != {len(phone_list)}"
        )

        # 构建三语 BERT 张量
        if self.zh_bert_extra:
            zh_bert = bert_feature
            ja_bert, en_bert = torch.zeros(self.ja_bert_dim, len(phone_list)), torch.zeros(1024, len(phone_list))
        elif self.ja_bert_extra:
            ja_bert = bert_feature
            zh_bert, en_bert = torch.zeros(1024, len(phone_list)), torch.zeros(1024, len(phone_list))
        elif language_str == "zh":
            zh_bert = bert_feature
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone_list))
            en_bert = torch.zeros(1024, len(phone_list))
        elif language_str == "ja":
            zh_bert = torch.zeros(1024, len(phone_list))
            ja_bert = bert_feature
            en_bert = torch.zeros(1024, len(phone_list))
        elif language_str == "en":
            zh_bert = torch.zeros(1024, len(phone_list))
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone_list))
            en_bert = bert_feature
        else:
            zh_bert = torch.zeros(1024, len(phone_list))
            ja_bert = torch.zeros(self.ja_bert_dim, len(phone_list))
            en_bert = torch.zeros(1024, len(phone_list))

        phone_t = torch.LongTensor(phone_list)
        tone_t = torch.LongTensor(tone_list)
        language_t = torch.LongTensor(language_list)

        return zh_bert, ja_bert, en_bert, phone_t, tone_t, language_t

    @property
    def pinyin_g2pw(self):
        """G2PW 多音字模型"""
        return self.model_handler.get_pinyin_g2pw()

    def _get_bert_feature_onnx(
        self, norm_text: str, word2ph: List[int], language: str,
        bert_model_name: str, style_text=None, style_weight=0.7
    ) -> torch.Tensor:
        """
        使用 ONNX BERT 获取文本特征，接口与原 model_handler.get_bert_feature 一致。
        """
        # 使用 model_handler 中的 lang-specific 函数处理 tokenization 和后处理，
        # 但用 ONNX 推理替代 HuggingFace 模型 forward。
        tokenizer, py_model = self.model_handler.get_bert_model(bert_model_name)

        # 获取对应语言的 BERT 处理函数
        lang_bert_func = self.model_handler.lang_bert_func_map.get(language)
        if lang_bert_func is None:
            raise ValueError(f"不支持的语言/BERT 组合: {language}")

        # 将 PyTorch BERT 模型临时替换为 ONNX wrapper，运行 lang-specific 函数
        ort_bert = _ONNXBertWrapper(
            language, bert_model_name, self, tokenizer, py_model
        )

        # 临时替换 model_handler 中的 BERT 模型
        original_model = self.model_handler.bert_models[bert_model_name]
        self.model_handler.bert_models[bert_model_name] = (tokenizer, ort_bert, 999)

        try:
            bert_feature = lang_bert_func(
                norm_text, word2ph, tokenizer, ort_bert,
                self.model_handler.device,
                style_text=style_text, style_weight=style_weight
            )
        finally:
            # 恢复原 PyTorch 模型
            self.model_handler.bert_models[bert_model_name] = original_model

        return bert_feature


class _ONNXBertWrapper:
    """
    将 ONNX BERT 包装为与 HuggingFace 模型兼容的接口。
    只包装训练期间 BERT 函数使用的属性。
    """

    def __init__(self, lang: str, bert_name: str, parent: BertVITS2ONNX,
                 tokenizer, py_model):
        self._lang = lang
        self._bert_name = bert_name
        self._parent = parent
        self.config = py_model.config
        # 转移 PyTorch 模型的必要属性
        if hasattr(py_model, "roberta"):
            self.roberta = py_model.roberta
        if hasattr(py_model, "bert"):
            self.bert = py_model.bert
        if hasattr(py_model, "deberta"):
            self.deberta = py_model.deberta

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        """模拟 HuggingFace 模型 forward，返回类似 BaseModelOutput 的对象"""
        import torch

        # 转为 numpy
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy()
        else:
            input_ids_np = np.asarray(input_ids)

        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask_np = attention_mask.cpu().numpy()
        elif attention_mask is not None:
            attention_mask_np = np.asarray(attention_mask)
        else:
            attention_mask_np = np.ones_like(input_ids_np)

        if token_type_ids is not None and isinstance(token_type_ids, torch.Tensor):
            token_type_ids_np = token_type_ids.cpu().numpy()
        elif token_type_ids is not None:
            token_type_ids_np = np.asarray(token_type_ids)
        else:
            token_type_ids_np = None

        # ONNX 推理
        hidden = self._parent._onnx_bert_forward(
            self._lang, input_ids_np, attention_mask_np, token_type_ids_np
        )

        # 包装为类似 HuggingFace 输出的对象
        return _FakeBertOutput(torch.from_numpy(hidden))

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self, mode=False):
        return self


class _FakeBertOutput:
    """模拟 HuggingFace BaseModelOutputWithPooling 的 last_hidden_state"""
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, idx):
        return self.last_hidden_state[idx]


# ------------------------------------------------------------------
# SynthesizerTrn ONNX 推理
# ------------------------------------------------------------------

def _onnx_synthesizer_infer(
    self: BertVITS2ONNX,
    sid: int,
    phones: torch.Tensor,
    tones: torch.Tensor,
    lang_ids: torch.Tensor,
    zh_bert: torch.Tensor,
    ja_bert: torch.Tensor,
    en_bert: torch.Tensor,
    sdp_ratio: float,
    noise: float,
    noisew: float,
    length: float,
    emo: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """通过 ONNX Runtime 运行 SynthesizerTrn 推理"""

    x = phones.unsqueeze(0).numpy().astype(np.int64)
    x_lengths = np.array([phones.size(0)], dtype=np.int64)
    sid_np = np.array([sid], dtype=np.int64)
    tone_np = tones.unsqueeze(0).numpy().astype(np.int64)
    lang_np = lang_ids.unsqueeze(0).numpy().astype(np.int64)

    # BERT 特征（加 batch 维）
    if self.zh_bert_extra:
        zh_np = zh_bert.unsqueeze(0).numpy().astype(np.float32)
        ja_np = np.zeros((1, self.ja_bert_dim, zh_np.shape[2]), dtype=np.float32)
        en_np = np.zeros((1, 1024, zh_np.shape[2]), dtype=np.float32)
    elif self.ja_bert_extra:
        ja_np = ja_bert.unsqueeze(0).numpy().astype(np.float32)
        zh_np = np.zeros((1, 1024, ja_np.shape[2]), dtype=np.float32)
        en_np = np.zeros((1, 1024, ja_np.shape[2]), dtype=np.float32)
    else:
        zh_np = zh_bert.unsqueeze(0).numpy().astype(np.float32)
        ja_np = ja_bert.unsqueeze(0).numpy().astype(np.float32)
        en_np = en_bert.unsqueeze(0).numpy().astype(np.float32)

    # 标量参数
    feed = {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid_np,
        "tone": tone_np,
        "language": lang_np,
        "zh_bert": zh_np,
        "ja_bert": ja_np,
        "en_bert": en_np,
        "noise_scale": np.array(noise, dtype=np.float32),
        "length_scale": np.array(length, dtype=np.float32),
        "noise_scale_w": np.array(noisew, dtype=np.float32),
        "sdp_ratio": np.array(sdp_ratio, dtype=np.float32),
    }

    if "emo" in self.syn_inputs and emo is not None:
        feed["emo"] = emo.numpy().astype(np.float32)

    feed = {k: v for k, v in feed.items() if k in self.syn_inputs}

    outputs = self.syn_session.run(None, feed)
    return outputs[0][0, 0]  # [t_audio] float32 numpy

# 挂载到类上
BertVITS2ONNX._onnx_synthesizer_infer = _onnx_synthesizer_infer


# ------------------------------------------------------------------
# 推理入口
# ------------------------------------------------------------------

def infer(
    self: BertVITS2ONNX,
    text: str,
    id: int,
    lang: List[str],
    sdp_ratio: float,
    noise: float,
    noisew: float,
    length: float,
    reference_audio=None,
    emotion=None,
    text_prompt=None,
    style_text=None,
    style_weight=0.7,
    **kwargs,
) -> np.ndarray:
    """
    Bert-VITS2 推理入口（与 Bert_VITS2.infer() 相同接口）。
    """
    language = lang[0]
    zh_bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text(
        text, language, style_text, style_weight
    )

    emo = None
    emotion_embedding = self.syn_meta.get("emotion_embedding", None) or getattr(
        self.hps_ms.model, "emotion_embedding", None
    )

    if emotion_embedding == 1:
        emo = self._get_emo(reference_audio, emotion)
    elif emotion_embedding == 2:
        emo = self._get_clap(reference_audio, text_prompt)

    return self._infer(
        id, phones, tones, lang_ids, zh_bert, ja_bert, en_bert,
        sdp_ratio, noise, noisew, length, emo
    )


def infer_multilang(
    self: BertVITS2ONNX,
    text: str,
    id: int,
    lang: List[str],
    sdp_ratio: float,
    noise: float,
    noisew: float,
    length: float,
    reference_audio=None,
    emotion=None,
    text_prompt=None,
    style_text=None,
    style_weight=0.7,
    **kwargs,
) -> np.ndarray:
    """多语言混合推理（与 Bert_VITS2.infer_multilang() 相同接口）"""
    from utils.sentence import split_languages

    target_languages = lang
    if len(lang) == 1 and lang[0] == "auto":
        target_languages = self.lang

    sentences_list = split_languages(
        text, target_languages=target_languages,
        expand_abbreviations=True, expand_hyphens=True
    )

    emo = None
    emotion_embedding = self.syn_meta.get("emotion_embedding", None) or getattr(
        self.hps_ms.model, "emotion_embedding", None
    )
    if emotion_embedding == 1:
        emo = self._get_emo(reference_audio, emotion)
    elif emotion_embedding == 2:
        emo = self._get_clap(reference_audio, text_prompt)

    phones_list, tones_list, lang_ids_list = [], [], []
    zh_bert_list, ja_bert_list, en_bert_list = [], [], []

    for idx, (_text, _lang) in enumerate(sentences_list):
        skip_start = idx != 0
        skip_end = idx != len(sentences_list) - 1
        _zh_bert, _ja_bert, _en_bert, _phones, _tones, _lang_ids = self.get_text(
            _text, _lang, style_text, style_weight
        )

        if skip_start:
            _phones = _phones[3:]
            _tones = _tones[3:]
            _lang_ids = _lang_ids[3:]
            _zh_bert = _zh_bert[:, 3:]
            _ja_bert = _ja_bert[:, 3:]
            _en_bert = _en_bert[:, 3:]
        if skip_end:
            _phones = _phones[:-2]
            _tones = _tones[:-2]
            _lang_ids = _lang_ids[:-2]
            _zh_bert = _zh_bert[:, :-2]
            _ja_bert = _ja_bert[:, :-2]
            _en_bert = _en_bert[:, :-2]

        phones_list.append(_phones)
        tones_list.append(_tones)
        lang_ids_list.append(_lang_ids)
        zh_bert_list.append(_zh_bert)
        ja_bert_list.append(_ja_bert)
        en_bert_list.append(_en_bert)

    zh_bert = torch.cat(zh_bert_list, dim=1)
    ja_bert = torch.cat(ja_bert_list, dim=1)
    en_bert = torch.cat(en_bert_list, dim=1)
    phones = torch.cat(phones_list, dim=0)
    tones = torch.cat(tones_list, dim=0)
    lang_ids = torch.cat(lang_ids_list, dim=0)

    return self._infer(
        id, phones, tones, lang_ids, zh_bert, ja_bert, en_bert,
        sdp_ratio, noise, noisew, length, emo
    )


def _infer(
    self: BertVITS2ONNX,
    id: int,
    phones: torch.Tensor,
    tones: torch.Tensor,
    lang_ids: torch.Tensor,
    zh_bert: torch.Tensor,
    ja_bert: torch.Tensor,
    en_bert: torch.Tensor,
    sdp_ratio: float,
    noise: float,
    noisew: float,
    length: float,
    emo: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """实际推理调度，优先 ONNX，失败时回落 PyTorch"""
    try:
        return self._onnx_synthesizer_infer(
            id, phones, tones, lang_ids, zh_bert, ja_bert, en_bert,
            sdp_ratio, noise, noisew, length, emo
        )
    except Exception as e:
        logger.warning(f"ONNX 推理失败 ({e})，使用 PyTorch 回落")
        return self._pytorch_synthesizer_fallback(
            id, phones, tones, lang_ids, zh_bert, ja_bert, en_bert,
            sdp_ratio, noise, noisew, length, emo
        )


def _pytorch_synthesizer_fallback(
    self: BertVITS2ONNX,
    id: int,
    phones: torch.Tensor,
    tones: torch.Tensor,
    lang_ids: torch.Tensor,
    zh_bert: torch.Tensor,
    ja_bert: torch.Tensor,
    en_bert: torch.Tensor,
    sdp_ratio: float,
    noise: float,
    noisew: float,
    length: float,
    emo: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """PyTorch CPU 回落推理"""
    from bert_vits2.bert_vits2 import Bert_VITS2

    if not hasattr(self, "_pytorch_model"):
        logger.info("初始化 PyTorch 回落模型...")
        self._pytorch_model = Bert_VITS2(
            self.vits_path, self.hps_ms, device=torch.device("cpu")
        )
        self._pytorch_model.load_model(self.model_handler)
        logger.info("PyTorch 回落模型就绪")

    return self._pytorch_model._infer(
        id, phones, tones, lang_ids, zh_bert, ja_bert, en_bert,
        sdp_ratio, noise, noisew, length, emo
    )


def _get_emo(self: BertVITS2ONNX, reference_audio, emotion):
    """获取情感向量（复用 model_handler）"""
    from bert_vits2.get_emo import get_emo

    if reference_audio is not None:
        emo = torch.from_numpy(
            get_emo(reference_audio, self.model_handler.emotion_model,
                    self.model_handler.emotion_processor)
        )
        return emo.unsqueeze(0)
    else:
        if emotion is None:
            emotion = 0
        return torch.Tensor([emotion]).unsqueeze(0)


def _get_clap(self: BertVITS2ONNX, reference_audio, text_prompt):
    """获取 CLAP 情感向量（复用 model_handler）"""
    import numpy as np

    from bert_vits2.clap_wrapper import get_clap_audio_feature, get_clap_text_feature
    from config import config as app_config

    if isinstance(reference_audio, np.ndarray):
        emo = get_clap_audio_feature(
            reference_audio, self.model_handler.clap_model,
            self.model_handler.clap_processor, self.device
        )
    else:
        if text_prompt is None:
            text_prompt = app_config.bert_vits2_config.text_prompt
        emo = get_clap_text_feature(
            text_prompt, self.model_handler.clap_model,
            self.model_handler.clap_processor, self.device
        )
    return torch.squeeze(emo, dim=1).unsqueeze(0)


# 挂载方法到类
BertVITS2ONNX.infer = infer
BertVITS2ONNX.infer_multilang = infer_multilang
BertVITS2ONNX._infer = _infer
BertVITS2ONNX._pytorch_synthesizer_fallback = _pytorch_synthesizer_fallback
BertVITS2ONNX._get_emo = _get_emo
BertVITS2ONNX._get_clap = _get_clap
