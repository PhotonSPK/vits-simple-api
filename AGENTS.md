# AGENTS.md — vits-simple-api 项目指南

本文档供 AI 编码助手（Copilot、Cursor、Codex 等）在处理本代码库时参考，
帮助 Agent 快速理解项目结构、架构约定和开发规范。

---

## 项目概述

**vits-simple-api** 是一个基于 Flask 的文字转语音（TTS）REST API 服务，
对多种深度学习 TTS 模型进行统一封装，并提供 Web 管理界面。

- **Python 版本**：3.10
- **默认端口**：23456
- **主要入口**：`app.py`
- **运行方式**：`python app.py` 或 `gunicorn -c gunicorn_config.py app:app`

---

## 支持的模型类型

在 `contants.py` 中以 `ModelType` 枚举定义：

| 枚举值 | 说明 |
|---|---|
| `VITS` | 原版 VITS 语音合成 / 语音转换 |
| `HUBERT_VITS` | HuBert-soft VITS 模型 |
| `W2V2_VITS` | W2V2-VITS / emotional-vits 维度情感模型 |
| `BERT_VITS2` | Bert-VITS2（支持多语言、情感、风格） |
| `GPT_SOVITS` | GPT-SoVITS（参考音频克隆） |

---

## 目录结构

```
vits-simple-api/
├── app.py                  # Flask 应用工厂，注册蓝图、调度任务
├── config.py               # Pydantic 配置模型，加载 config.yaml
├── contants.py             # ModelType 枚举定义
├── gunicorn_config.py      # Gunicorn 生产部署配置
├── logger.py               # 日志初始化
│
├── manager/                # 模型管理核心层
│   ├── ModelManager.py     # 模型加载/卸载，实现 Subject（观察者模式）
│   ├── TTSManager.py       # TTS 推理调度，实现 Observer（观察者模式）
│   ├── model_handler.py    # 模型文件扫描与路径解析
│   └── observer.py         # Observer/Subject 抽象基类
│
├── tts_app/                # Flask 应用层（蓝图）
│   ├── model_manager.py    # 单例：初始化 ModelManager 和 TTSManager
│   ├── voice_api/          # /voice/* 语音合成 API 蓝图
│   │   ├── views.py        # 所有 API 路由
│   │   ├── auth.py         # API Key 鉴权装饰器
│   │   └── utils/          # 音频处理工具函数
│   ├── admin/              # /admin 管理后台蓝图（需登录）
│   │   └── views.py        # 模型加载/卸载、配置保存
│   ├── auth/               # 登录/登出蓝图
│   ├── frontend/           # 前端页面蓝图
│   ├── templates/          # Jinja2 HTML 模板
│   └── static/             # 静态资源
│
├── vits/                   # VITS / HuBert-VITS / W2V2-VITS 模型实现
│   ├── vits.py
│   ├── hubert_vits.py
│   └── w2v2_vits.py
│
├── bert_vits2/             # Bert-VITS2 模型实现
│   └── bert_vits2.py
│
├── gpt_sovits/             # GPT-SoVITS 模型实现
│   └── gpt_sovits.py
│
├── utils/                  # 通用工具
│   ├── classify_language.py  # 语言识别（fastlid / langid）
│   ├── sentence.py           # 长文本分段、语言标记
│   ├── data_utils.py         # HParams、文件清理等
│   ├── phrases_dict.py       # 词语词典初始化
│   └── lang_dict.py          # 语言代码映射
│
├── module/                 # 多音字、G2pW 等文本前端模块
├── data/                   # 运行时数据（BERT 模型、TTS 模型、参考音频等）
├── docker-compose.yaml
├── docker-compose-gpu.yaml
└── requirements.txt
```

---

## 架构模式

### 观察者模式（Observer Pattern）

`ModelManager`（Subject）与 `TTSManager`（Observer）解耦：

- 模型加载/卸载后，`ModelManager` 调用 `notify()` 通知所有观察者。
- `TTSManager.update()` 响应 `"model_loaded"` / `"model_unloaded"` 事件，
  更新内部模型引用，无需重启服务。

```
ModelManager ──notify()──► TTSManager.update()
```

### 单例初始化

`tts_app/model_manager.py` 在模块级别完成单例构建：

```python
model_manager = ModelManager(config.system.device)
tts_manager = TTSManager(model_manager)
model_manager.attach(tts_manager)
model_manager.model_init()
```

所有蓝图通过 `from tts_app.model_manager import model_manager, tts_manager` 共享同一实例。

---

## API 路由速览

所有语音接口挂载在 `/voice/` 前缀下，支持 `GET` 和 `POST`（JSON / Form）。

| 路由 | 说明 | 需要 API Key |
|---|---|---|
| `GET /voice/speakers` | 列出所有已加载模型的说话人 | 否 |
| `GET /voice/default_parameter` | 查询各模型默认参数 | 否 |
| `GET/POST /voice/vits` | VITS 合成 | 是 |
| `GET/POST /voice/hubert-vits` | HuBert-VITS 声音转换 | 是 |
| `GET/POST /voice/w2v2-vits` | W2V2-VITS 情感合成 | 是 |
| `GET/POST /voice/bert-vits2` | Bert-VITS2 合成 | 是 |
| `GET/POST /voice/gpt-sovits` | GPT-SoVITS 合成 | 是 |
| `GET/POST /voice/convert` | 语音转换 | 是 |
| `GET/POST /voice/ssml` | SSML 合成 | 是 |
| `GET/POST /voice/dimension_emotion` | 情感向量提取 | 是 |

---

## 配置系统

配置由 `config.py` 中的 Pydantic 模型读取 `config.yaml` 实现，主要配置节：

| 配置节 | 说明 |
|---|---|
| `system` | 设备（cpu/cuda）、端口、数据路径、管理后台开关 |
| `http_service` | 允许的 CORS origins、API Key 设置 |
| `tts_model_config` | 模型目录、自动扫描开关、模型列表 |
| `vits_config` | VITS 默认参数（id、format、lang、length、noise 等） |
| `w2v2_vits_config` | W2V2-VITS 默认参数 |
| `hubert_vits_config` | HuBert-VITS 默认参数 |
| `bert_vits2_config` | Bert-VITS2 默认参数（含 sdp_ratio、emotion、style） |
| `gpt_sovits_config` | GPT-SoVITS 默认参数及 presets（参考音频预设） |
| `language_identification` | 语言识别库（fastlid/langid）、分词正则、自动检测范围 |

修改配置后无需重启，管理后台的"保存配置"接口会调用 `save_config_to_yaml()` 持久化。

---

## 音频输出格式

支持格式：`wav`、`mp3`、`ogg`、`flac`、`silk`（微信语音）。

- 流式响应（`streaming=true`）**仅支持 MP3**，传入其他格式会被自动降级。
- 编码在 `TTSManager.encode()` 中集中处理，使用 `soundfile` + `silkcoder`。

---

## 语言处理流程

1. **语言识别**：`utils/classify_language.py` 使用 `fastlid` 或 `langid` 对文本片段分类。
2. **文本分段**：`utils/sentence.py` 中的 `split_languages()` 和 `markup_language()`
   将混合语言文本拆分为带语言标签的片段列表，再交给对应的文本前端处理。
3. **多音字处理**：`module/polyphonic/` 在应用启动时预加载。
4. **词语词典**：`utils/phrases_dict.py` 中的 `phrases_dict_init()` 在启动时初始化。

---

## 开发约定

### 新增模型类型

1. 在 `contants.py` 的 `ModelType` 中添加枚举值。
2. 在对应目录（如 `new_model/`）实现模型类。
3. 在 `config.py` 添加对应的 Pydantic 配置模型。
4. 在 `manager/ModelManager.py` 的 `model_class_map` 中注册。
5. 在 `manager/TTSManager.py` 的 `infer_map` 中注册推理方法。
6. 在 `tts_app/voice_api/views.py` 中添加路由。

### 新增 API 路由

- 路由统一在 `tts_app/voice_api/views.py` 中使用 `@voice_api.route()` 装饰器定义。
- 需要鉴权的接口加 `@require_api_key` 装饰器（来自 `tts_app/voice_api/auth.py`）。
- 参数统一通过 `get_param(request_data, key, default, data_type)` 获取，支持 GET/POST/JSON。

### 管理接口（需登录）

- 管理后台接口统一在 `tts_app/admin/views.py` 中，使用 `@login_required` 保护。
- 登录逻辑在 `tts_app/auth/views.py`，基于 Flask-Login。

---

## 部署

### 本地运行

```bash
pip install -r requirements.txt
python app.py
```

### Docker

```bash
# CPU 版
docker-compose up -d

# GPU 版
docker-compose -f docker-compose-gpu.yaml up -d
```

### Gunicorn 生产部署

```bash
gunicorn -c gunicorn_config.py app:app
```

Gunicorn 配置说明（`gunicorn_config.py`）：
- `workers = 1`（TTS 模型为内存密集型，多 worker 会 OOM）
- `preload_app = True` + GC freeze 优化 fork 后的内存占用。

---

## 数据目录约定（`data/`）

| 子目录 | 用途 |
|---|---|
| `data/models/` | TTS 模型文件（`.pth`、`.onnx`、config JSON） |
| `data/bert/` | BERT/DeBERTa 等预训练语言模型 |
| `data/emotional/` | 情感模型（CLAP、wav2vec2 等） |
| `data/hubert/` | HuBert 特征提取模型 |
| `data/reference_audio/` | GPT-SoVITS 参考音频 |
| `data/G2PWModel/` | G2pW 中文多音字模型 |

---

## 关键依赖

| 包 | 用途 |
|---|---|
| `flask` | Web 框架 |
| `flask-login` | 管理后台身份验证 |
| `flask-wtf` | CSRF 保护 |
| `flask-cors` | 跨域资源共享 |
| `flask-apscheduler` | 定时清理临时文件 |
| `torch` | 深度学习推理后端 |
| `transformers` | BERT 系列模型加载 |
| `soundfile` / `librosa` | 音频 I/O 与处理 |
| `graiax-silkcoder` | SILK 格式编码（微信语音） |
| `langid` / `fastlid` | 语言自动识别 |
| `pydantic` | 配置验证与序列化 |
