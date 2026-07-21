"""
Bert-VITS2 ONNX 导出脚本
=======================
将 Bert-VITS2 的 SynthesizerTrn 和 BERT 模型导出为 ONNX 格式。

用法:
    # 导出 SynthesizerTrn（VITS 解码器）
    python -m bert_vits2.onnx_export synthesize \
        --model-path data/models/your_model.pth \
        --config-path data/models/your_model/config.json \
        --output-dir data/models/your_model/

    # 导出 BERT 模型
    python -m bert_vits2.onnx_export bert \
        --model-name CHINESE_ROBERTA_WWM_EXT_LARGE \
        --output-dir data/models/your_model/

    # 导出 CLAP 情感模型
    python -m bert_vits2.onnx_export clap \
        --model-path data/emotional/clap-htsat-fused \
        --output-dir data/models/your_model/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到 path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bert_vits2 import commons
from bert_vits2.models import SynthesizerTrn
from bert_vits2.models_v230 import SynthesizerTrn as SynthesizerTrn_v230
from bert_vits2.models_ja_extra import SynthesizerTrn as SynthesizerTrn_ja_extra
from bert_vits2.utils import process_legacy_versions
from utils import get_hparams_from_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SynthesizerTrn ONNX Wrapper — 去除条件分支，适合 torch.onnx.export 追踪
# ---------------------------------------------------------------------------

class SynthesizerTrnTracable(nn.Module):
    """
    将 SynthesizerTrn.infer() 包装为纯前向函数，
    speaker embedding (emb_g) 内置于模型中，以 sid 作为输入（不是 g）。
    """

    def __init__(self, synthesizer: nn.Module):
        super().__init__()
        self.enc_p = synthesizer.enc_p
        self.dec = synthesizer.dec
        self.flow = synthesizer.flow
        self.sdp = synthesizer.sdp
        self.dp = synthesizer.dp
        self.n_speakers = synthesizer.n_speakers

        # 将 emb_g 融入可追踪模型
        if hasattr(synthesizer, "emb_g"):
            self.emb_g = synthesizer.emb_g
        else:
            gin_channels = getattr(synthesizer, "gin_channels", 256)
            self.emb_g = nn.Embedding(max(1, synthesizer.n_speakers), gin_channels)

    def forward(
        self,
        x,             # [1, t_x] int64 — phones
        x_lengths,     # [1] int64
        sid,           # [1] int64 — speaker id
        tone,          # [1, t_x] int64
        language,      # [1, t_x] int64
        zh_bert,       # [1, 1024, t_x] float32
        ja_bert,       # [1, ja_bert_dim, t_x] float32
        en_bert,       # [1, 1024, t_x] float32
        noise_scale,   # float32 scalar
        length_scale,  # float32 scalar
        noise_scale_w, # float32 scalar
        sdp_ratio,     # float32 scalar
        emo=None,      # [1, emo_dim] float32, optional
    ):
        # Speaker embedding（内置于 ONNX 图）
        g = self.emb_g(sid).unsqueeze(-1)  # [1, gin_channels, 1]

        # 时长预测（SDP + DP 混合）
        logw = (
            self.sdp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio
            + self.dp(x_enc, x_mask, g=g) * (1 - sdp_ratio)
        )
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p_upsampled = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_upsampled = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # 随机噪声（ONNX RandomNormal）
        z_p = m_p_upsampled + torch.randn_like(m_p_upsampled) * torch.exp(logs_p_upsampled) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o


# ---------------------------------------------------------------------------
# 导出 SynthesizerTrn
# ---------------------------------------------------------------------------

def export_synthesizer(model_path: str, config_path: str, output_dir: str, opset: int = 17):
    """导出 VITS 解码器到 ONNX"""
    logger.info(f"加载配置: {config_path}")
    hps = get_hparams_from_file(config_path)

    # 确定模型版本和对应的类
    version = process_legacy_versions(hps).lower().replace("-", "_")
    if version in ["2.3", "extra", "2.4"]:
        SynthClass = SynthesizerTrn_v230
    elif version == "ja_extra":
        SynthClass = SynthesizerTrn_ja_extra
    else:
        SynthClass = SynthesizerTrn

    # 解析模型参数
    symbols = getattr(hps, "symbols", None)
    if symbols is None:
        from bert_vits2.text import symbols as syms
        symbols = syms

    ja_bert_dim = 1024
    if hasattr(hps.data, "lang"):
        if "ja" in hps.data.lang:
            if version in ["1.0", "1.0.0", "1.0.1", "1.1", "1.1.0", "1.1.1", "1.1.0_transition"]:
                ja_bert_dim = 768

    zh_bert_extra = version in ["extra", "2.4"]
    num_tones = None
    try:
        from bert_vits2.text import num_tones as nt
        num_tones = nt
    except ImportError:
        pass

    emotion_embedding = getattr(hps.model, "emotion_embedding", None)

    # 创建模型
    n_speakers = getattr(hps.data, "n_speakers", 0)
    model = SynthClass(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=n_speakers,
        symbols=symbols,
        ja_bert_dim=ja_bert_dim,
        num_tones=num_tones,
        zh_bert_extra=zh_bert_extra,
        **hps.model,
    )
    model.eval()

    # 加载权重
    logger.info(f"加载模型权重: {model_path}")
    from bert_vits2 import utils as bert_vits2_utils
    bert_vits2_utils.load_checkpoint(model_path, model, None, skip_optimizer=True, version=version)

    # 提取 gin_channels
    gin_channels = getattr(hps.model, "inter_channels", 192)

    # 包装为可追踪模型
    traceable = SynthesizerTrnTracable(model)
    traceable.eval()

    # 准备示例输入
    t_x = 50  # 示例文本长度
    emo_dim = 1024 if emotion_embedding == 1 else (512 if emotion_embedding == 2 else None)

    dummy_x = torch.randint(0, len(symbols), (1, t_x), dtype=torch.long)
    dummy_x_lengths = torch.tensor([t_x], dtype=torch.long)
    dummy_sid = torch.tensor([0], dtype=torch.long)
    dummy_tone = torch.randint(0, num_tones or 10, (1, t_x), dtype=torch.long)
    dummy_lang = torch.zeros(1, t_x, dtype=torch.long)
    dummy_zh_bert = torch.randn(1, 1024, t_x, dtype=torch.float32)
    dummy_ja_bert = torch.randn(1, ja_bert_dim, t_x, dtype=torch.float32)
    dummy_en_bert = torch.randn(1, 1024, t_x, dtype=torch.float32)
    dummy_noise = torch.tensor(0.667, dtype=torch.float32)
    dummy_length = torch.tensor(1.0, dtype=torch.float32)
    dummy_noisew = torch.tensor(0.8, dtype=torch.float32)
    dummy_sdp = torch.tensor(0.2, dtype=torch.float32)

    # 动态轴名称
    dynamic_axes = {
        "x": {1: "t_x"},
        "tone": {1: "t_x"},
        "language": {1: "t_x"},
        "zh_bert": {2: "t_x"},
        "ja_bert": {2: "t_x"},
        "en_bert": {2: "t_x"},
        "audio": {2: "t_audio"},
    }

    input_names = [
        "x", "x_lengths", "sid", "tone", "language",
        "zh_bert", "ja_bert", "en_bert",
        "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio",
    ]
    inputs = (
        dummy_x, dummy_x_lengths, dummy_sid, dummy_tone, dummy_lang,
        dummy_zh_bert, dummy_ja_bert, dummy_en_bert,
        dummy_noise, dummy_length, dummy_noisew, dummy_sdp,
    )

    if emo_dim is not None:
        dummy_emo = torch.randn(1, emo_dim, dtype=torch.float32)
        input_names.append("emo")
        inputs = inputs + (dummy_emo,)
        dynamic_axes["emo"] = {1: "emo_dim"}

    # 导出
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthesizer.onnx")
    logger.info(f"导出 SynthesizerTrn → {output_path}")

    torch.onnx.export(
        traceable,
        inputs,
        output_path,
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info("SynthesizerTrn 导出完成 ✓")

    # 保存元数据
    meta = {
        "version": version,
        "n_speakers": n_speakers,
        "gin_channels": gin_channels,
        "ja_bert_dim": ja_bert_dim,
        "zh_bert_extra": zh_bert_extra,
        "num_tones": num_tones,
        "emotion_embedding": emotion_embedding,
        "emo_dim": emo_dim,
        "sampling_rate": hps.data.sampling_rate,
        "symbols": symbols,
    }
    meta_path = os.path.join(output_dir, "synthesizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"保存元数据 → {meta_path}")


# ---------------------------------------------------------------------------
# 导出 BERT 模型
# ---------------------------------------------------------------------------

def export_bert(model_name: str, model_path: str, output_dir: str, opset: int = 17):
    """导出 HuggingFace BERT 模型到 ONNX"""
    from transformers import AutoModelForMaskedLM, AutoTokenizer, MegatronBertModel, BertTokenizer

    logger.info(f"加载 BERT 模型: {model_name} from {model_path}")

    if model_name == "Erlangshen_MegatronBert_1.3B_Chinese":
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = MegatronBertModel.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)

    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"bert_{model_name}.onnx")

    # 示例输入
    seq_len = 128
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    if model_name == "Erlangshen_MegatronBert_1.3B_Chinese":
        # MegatronBert 不需要 token_type_ids
        logger.info(f"导出 BERT → {output_path}")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "last_hidden_state": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    else:
        dummy_token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
        logger.info(f"导出 BERT → {output_path}")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "token_type_ids": {0: "batch", 1: "seq_len"},
                "last_hidden_state": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    logger.info(f"BERT 导出完成 ✓ → {output_path}")


# ---------------------------------------------------------------------------
# 导出 CLAP 情感模型
# ---------------------------------------------------------------------------

def export_clap(model_path: str, output_dir: str, opset: int = 17):
    """导出 CLAP 模型到 ONNX"""
    from transformers import ClapModel

    logger.info(f"加载 CLAP 模型: {model_path}")
    model = ClapModel.from_pretrained(model_path)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clap.onnx")

    # 示例音频输入
    batch = 1
    audio_len = 480000  # 30s @ 16kHz, 将通过动态轴支持变化
    dummy_input = torch.randn(batch, audio_len, dtype=torch.float32)

    logger.info(f"导出 CLAP → {output_path}")
    # CLAP 的音频编码器部分
    audio_encoder = model.audio_model

    torch.onnx.export(
        audio_encoder,
        dummy_input,
        output_path,
        input_names=["audio_input"],
        output_names=["audio_embedding"],
        dynamic_axes={
            "audio_input": {0: "batch", 1: "audio_len"},
            "audio_embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info(f"CLAP 导出完成 ✓ → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bert-VITS2 ONNX 导出工具")
    subparsers = parser.add_subparsers(dest="command", help="导出目标")

    # synthesize
    syn_parser = subparsers.add_parser("synthesize", help="导出 SynthesizerTrn")
    syn_parser.add_argument("--model-path", required=True, help="PyTorch 模型权重路径 (.pth)")
    syn_parser.add_argument("--config-path", required=True, help="模型配置路径 (config.json)")
    syn_parser.add_argument("--output-dir", required=True, help="ONNX 输出目录")
    syn_parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")

    # bert
    bert_parser = subparsers.add_parser("bert", help="导出 BERT 模型")
    bert_parser.add_argument("--model-name", required=True, help="BERT 模型名称")
    bert_parser.add_argument("--model-path", required=True, help="HuggingFace 模型路径")
    bert_parser.add_argument("--output-dir", required=True, help="ONNX 输出目录")
    bert_parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")

    # clap
    clap_parser = subparsers.add_parser("clap", help="导出 CLAP 模型")
    clap_parser.add_argument("--model-path", required=True, help="CLAP 模型路径")
    clap_parser.add_argument("--output-dir", required=True, help="ONNX 输出目录")
    clap_parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")

    args = parser.parse_args()

    if args.command == "synthesize":
        export_synthesizer(args.model_path, args.config_path, args.output_dir, args.opset)
    elif args.command == "bert":
        export_bert(args.model_name, args.model_path, args.output_dir, args.opset)
    elif args.command == "clap":
        export_clap(args.model_path, args.output_dir, args.opset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
