from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

CUSTOM_CSS = """
<style>
:root{--ink:#FAFAFA;--muted:#C7CBD4;--panel:#262730;--line:#454752;--accent:#FF4B4B;--accent-hover:#E03E3E;--code:#1E1E1E}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#0E1117}
[data-testid="stAppViewContainer"] *,[data-testid="stSidebar"] *{color:var(--ink)}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] [data-baseweb="select"]>div,div[data-baseweb="select"]>div,[role="option"]{background:#101a2d!important;color:var(--ink)!important;border:1px solid var(--line)!important;box-shadow:none!important}
[data-testid="stSidebar"] div[data-baseweb="select"],[data-testid="stSidebar"] div[data-baseweb="select"]>div{background-color:#101a2d!important;color:#e8eefc!important;border-color:var(--line)!important}
[data-testid="stSidebar"] div[data-baseweb="select"] span,[data-testid="stSidebar"] div[data-baseweb="select"] input{color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important}
[data-testid="stSidebar"] div[data-baseweb="select"] svg{fill:#e8eefc!important;color:#e8eefc!important}
[data-testid="stSidebar"] [role="option"]{background-color:#101a2d!important;color:#e8eefc!important}
[data-testid="stSidebar"] input:not([type="checkbox"]),[data-testid="stSidebar"] textarea{background-color:rgba(15,23,42,.92)!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;border-color:var(--line)!important}
[data-testid="stSidebar"] input:not([type="checkbox"])::placeholder,[data-testid="stSidebar"] textarea::placeholder{color:#93a2bf!important;-webkit-text-fill-color:#93a2bf!important}
[data-testid="stSidebar"] .stTextInput label,[data-testid="stSidebar"] .stTextArea label,[data-testid="stSidebar"] .stSelectbox label{color:#f8fafc!important;font-weight:500!important}
[data-testid="stSidebar"] .stTextInput div:has(> input),[data-testid="stSidebar"] .stTextArea div:has(> textarea){background-color:rgba(15,23,42,.92)!important}
[data-testid="stSidebar"] .stSelectbox div:has(> input[role="combobox"]){background-color:#101a2d!important}
.stSelectbox .react-aria-ComboBox>div,.stSelectbox div:has(> input[role="combobox"]){background-color:#101a2d!important;color:#e8eefc!important;border:1px solid var(--line)!important;border-radius:10px!important}
.stSelectbox input[role="combobox"]{background-color:transparent!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.stSelectbox [role="option"]{background-color:#101a2d!important;color:#e8eefc!important}
div[data-testid="stCode"],div[data-testid="stCode"]>div,div[data-testid="stCode"] pre,div[data-testid="stCode"] code,div[data-testid="stCodeBlock"],div[data-testid="stCodeBlock"]>div,div[data-testid="stCodeBlock"] pre,div[data-testid="stCodeBlock"] code,.stCodeBlock,.stCodeBlock>div,.stCodeBlock pre,.stCodeBlock code,[data-testid="stMarkdownContainer"] pre,[data-testid="stMarkdownContainer"] code{background:#1E1E1E!important;color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important;border-color:var(--line)!important}
div[data-testid="stCode"] pre *,div[data-testid="stCode"] code *,div[data-testid="stCodeBlock"] pre *,div[data-testid="stCodeBlock"] code *,.stCodeBlock pre *,.stCodeBlock code *,[data-testid="stMarkdownContainer"] pre *,[data-testid="stMarkdownContainer"] pre code{background:transparent!important;color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important}
.stTabs [role="tablist"]{background:#0a111f!important;overflow-x:auto!important;scrollbar-color:#50d9e9 #101a2d;white-space:nowrap;gap:.4rem}
.stTabs [role="tab"]{background:#101a2d!important;color:#cbd5e1!important;white-space:nowrap!important;padding:8px 16px!important;font-size:0.95rem!important;text-overflow:unset!important;overflow:visible!important}
.stTabs [role="tab"][aria-selected="true"]{background:#173b60!important;color:#ffffff!important;border-bottom:2px solid #50d9e9!important}
.stTabs [role="tab"]:hover{background:#1e293b!important}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;background:#0a111f8f;padding:.35rem;border-radius:13px;overflow-x:auto;scrollbar-width:thin}
.stTabs [data-baseweb="tab"]{border-radius:9px;color:var(--muted)!important;white-space:nowrap!important;text-align:center;min-height:2.5rem;padding:8px 16px!important;font-size:0.95rem!important}
.stSelectbox label,.stTextInput label,.stTextArea label{color:var(--ink)!important}
.stWarning,[data-baseweb="notification"],.stAlert,.stAlertContainer{background:rgba(234,179,8,0.12)!important;border:1px solid rgba(234,179,8,0.4)!important;color:#fde047!important;font-weight:500!important}
.stWarning > div,[data-baseweb="notification"] > div,.stAlert > div{color:#fde047!important}
.stWarning svg,[data-baseweb="notification"] svg,.stAlert svg{fill:#fde047!important;color:#fde047!important}
div[data-baseweb="radio"]{display:flex;gap:.35rem;overflow-x:auto;scrollbar-width:thin;padding-bottom:.25rem}
div[data-baseweb="radio"] > label{background:#101a2d!important;border:1px solid var(--line)!important;border-radius:9px!important;color:#cbd5e1!important;padding:8px 16px!important;font-size:0.95rem!important;white-space:nowrap!important;text-align:center!important;min-width:fit-content!important;cursor:pointer}
div[data-baseweb="radio"] > label > div[data-testid="stMarkdownContainer"],div[data-baseweb="radio"] > label > p{color:#cbd5e1!important}
div[data-baseweb="radio"] > label[data-checked="true"],div[data-baseweb="radio"] > label[aria-checked="true"]{background:#173b60!important;color:#ffffff!important;border-bottom:2px solid #50d9e9!important}
div[data-baseweb="radio"] > label:hover{background:#1e293b!important}
div[data-baseweb="radio"] input{display:none!important}
.stDownloadButton>button,.stButton>button{background:#262730!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:10px;font-weight:700;white-space:normal;line-height:1.2;height:48px;min-height:48px;overflow:hidden}
.stDownloadButton>button:hover,.stButton>button:hover{background:var(--accent-hover)!important;border-color:var(--accent-hover)!important;color:#fff!important}
[data-testid="stSidebar"]{background:rgba(7,13,25,.96);border-right:1px solid var(--line);min-width:18rem;max-width:22rem}
.block-container{width:min(100% - 2rem,1540px);max-width:1540px;padding-top:1.5rem!important;padding-bottom:2rem!important}
.hero{background:#262730!important;border:1px solid #454752!important;border-radius:20px;padding:clamp(1.2rem,3vw,2rem) clamp(1.1rem,3vw,2.3rem);margin-bottom:1.1rem;box-shadow:0 20px 55px #0004}
.hero h1{color:#f8fbff!important;margin:0 0 .55rem;font-size:clamp(1.55rem,3.4vw,2.35rem);line-height:1.12;overflow-wrap:anywhere;word-break:normal}
.hero p{color:#b9c8e2!important;margin:0;line-height:1.45}
.badge,.domain-badge{display:inline-block;padding:.3rem .68rem;border-radius:999px;font-size:.75rem;font-weight:750}
.badge{background:#262730!important;color:#FFFFFF!important;border:1px solid #454752}
.domain-badge{background:#FF4B4B!important;color:#FFFFFF!important;margin-top:.9rem}
.glass-card{background:#262730!important;border:1px solid #454752!important;border-radius:16px;padding:1.1rem;min-height:185px;box-shadow:0 14px 32px #0003;overflow-wrap:anywhere}
.glass-card h4{margin:0 0 .45rem;color:#f8fbff!important}.glass-card p,.glass-card small{color:#FAFAFA!important}
.callout{background:#262730!important;border:1px solid #454752!important;border-left:4px solid #FF4B4B!important;border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0 1rem;overflow-wrap:anywhere;color:#FAFAFA!important}
.callout strong{color:#FAFAFA!important}
[data-testid="stMetric"]{background:#262730!important;border:1px solid #454752!important;border-radius:15px;padding:.85rem 1rem;min-width:0!important;color:#FAFAFA!important}
[data-testid="stMetric"] label{color:#FAFAFA!important;white-space:normal!important;line-height:1.25!important;overflow:visible!important}
[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#FAFAFA!important;white-space:nowrap!important}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}
.stButton>button[kind="primary"]{background:var(--accent)!important;color:#fff!important;border:0;font-weight:800;height:48px;min-height:48px}
.pdf-open-link{display:flex;align-items:center;justify-content:center;min-height:3rem;padding:.5rem 1rem;border-radius:.5rem;background:#d95757;color:#fff!important;font-weight:800;text-decoration:none!important;text-align:center}
.pdf-open-link:hover{background:var(--accent-hover);color:#fff!important}
.stDataFrame,.stTable{width:100%;max-width:100%!important;overflow-x:auto!important;border-radius:12px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stDataFrame"]>div{max-width:100%!important;overflow-x:auto!important}
[data-testid="stImage"],[data-testid="stImage"]>div,[data-testid="stImage"] img{width:100%!important;max-width:100%!important;background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;padding:0!important}
[data-testid="stImage"]{margin:0 0 1rem!important}
[data-testid="stMarkdownContainer"] .katex-display{background:var(--code)!important;color:var(--ink)!important;padding:.6rem .9rem;border-radius:10px;margin:.6rem 0;overflow-x:auto}
[data-testid="stMarkdownContainer"] .katex{background:transparent!important;color:var(--ink)!important;padding:0 .12rem}
[data-testid="stExpander"]{border:1px solid #454752!important;border-radius:12px;background:#262730!important;color:#FAFAFA!important}
[data-testid="stExpander"] *{color:#FAFAFA!important}
.artifact-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1rem;min-width:0;overflow-wrap:anywhere}
.artifact-card strong{color:#f8fbff!important}.artifact-card small{color:var(--muted)!important}
.architecture-flow{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;background:#262730;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.25rem;margin:.35rem 0 1.15rem;box-shadow:0 12px 28px #0003}
.architecture-flow .flow-step{display:inline-flex;align-items:center;min-height:2.35rem;padding:.55rem .8rem;border:1px solid var(--line);border-radius:9px;background:#3A3B46;color:var(--ink)!important;font-weight:700;line-height:1.25}
.architecture-flow .flow-arrow{color:var(--accent)!important;font-size:1.25rem;font-weight:800}
.dataset-command{background:#1E1E1E!important;color:#FAFAFA!important;border:1px solid var(--line);border-radius:12px;padding:1.05rem 1.25rem;margin-top:.55rem;overflow-x:auto}
.dataset-command code{background:transparent!important;color:var(--ink)!important;font-family:Consolas,"Courier New",monospace;font-size:.95rem;white-space:pre}
.dataset-command .muted-command{color:var(--muted)!important}.dataset-command .path-command{color:#FF8A8A!important}.dataset-command .flag-command{color:#f59e0b!important}
.architecture-code{background:#1E1E1E!important;color:#FAFAFA!important;border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.35rem;margin-top:.55rem;overflow-x:auto}
.architecture-code pre{margin:0;background:transparent!important;color:var(--ink)!important;font-family:Consolas,"Courier New",monospace;font-size:.95rem;line-height:1.55;white-space:pre}
[data-testid="stMarkdownContainer"] pre{background:#1E1E1E!important;color:#FAFAFA!important;border:1px solid var(--line)!important;border-radius:12px!important;padding:1rem 1.15rem!important;font-family:Consolas,"Courier New",monospace!important;font-size:.9rem!important;line-height:1.5!important;white-space:pre!important;overflow-x:auto!important}
.chat-response{background:#262730!important;color:#FAFAFA!important;border:1px solid #363945!important;border-radius:8px;padding:1rem;margin-top:.75rem;line-height:1.55;overflow-wrap:anywhere}
.metric-card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:.85rem 1rem;min-height:132px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}
.metric-card-title{color:var(--muted);font-size:.92rem;font-weight:600;line-height:1.25;min-height:2.5rem;white-space:normal;overflow-wrap:anywhere;max-height:5rem;overflow:hidden;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical}
.metric-card-title span{display:block;color:var(--muted)}
.metric-card-value{color:#f8fbff;font-size:1.85rem;line-height:1.1;white-space:normal;overflow-wrap:anywhere}
.metric-card-badge{align-self:flex-start;max-width:100%;padding:.28rem .58rem;border-radius:999px;background:#1a5449;color:#d9fff2;font-size:.78rem;line-height:1.25;white-space:normal;overflow-wrap:anywhere}
@media (max-width: 760px){.block-container{width:calc(100% - 1rem);padding-left:.5rem;padding-right:.5rem}.hero h1{font-size:1.5rem}.stTabs [data-baseweb="tab"]{font-size:.78rem;padding:.35rem}.glass-card{min-height:0}.architecture-flow{gap:.35rem;padding:.8rem}.architecture-flow .flow-step{font-size:.82rem}.dataset-command{font-size:.8rem}}
</style>
"""

PRESETS = {
    "Dynamic Attention Head Gating for Efficient Transformer Inference": {
        "domain": "NLP / Efficiency",
        "objective": "Improve inference efficiency while preserving language modeling quality",
        "constraints": "CPU-friendly evaluation, transparent ablations, reproducible metrics",
    },
    "Vision Transformer with Multi-Scale Patch Embedding for CIFAR-10 & Medical Imaging": {
        "domain": "Computer Vision",
        "objective": "Improve visual representation quality across natural and medical images",
        "constraints": "Compare multi-scale features, report latency and robustness",
    },
    "Image Captioning with CLIP": {
        "domain": "Multimodal & Vision-Language",
        "objective": "Generate faithful image captions using CLIP-aligned visual and textual representations",
        "constraints": "Evaluate caption quality, cross-modal alignment, grounding, and hallucination risk",
    },
    "Spatio-Temporal Graph Neural Networks for Real-Time Financial Fraud Detection": {
        "domain": "Graph Learning / FinTech",
        "objective": "Detect suspicious transaction patterns with high recall and low latency",
        "constraints": "Imbalanced labels, temporal split, interpretable decisions",
    },
    "Contrastive Multimodal Learning for Automated Radiology Report Generation": {
        "domain": "Healthcare AI",
        "objective": "Align radiology images and clinical language for accurate report generation",
        "constraints": "Cross-modal alignment, clinical safety, external validation",
    },
}

CUSTOM = "Custom Topic (Enter any research topic...)"

DISPLAY_LABELS = {
    "model_name": "Model (Mô hình)",
    "name": "Model (Mô hình)",
    "role": "Role (Vai trò)",
    "backbone": "Backbone (Kiến trúc nền)",
    "primary_metric": "Primary Metric (Chỉ số chính)",
    "latency_ms": "Latency (Độ trễ, ms)",
    "vram_mb": "Peak VRAM (Bộ nhớ, MB)",
    "flops_g": "FLOPs (G)",
    "params_m": "Parameters (Tham số, M)",
    "dataset_name": "Dataset (Tập dữ liệu)",
    "evidence_type": "Evidence (Bằng chứng)",
    "configuration": "Configuration (Cấu hình)",
    "removed_component": "Removed Component (Thành phần bỏ)",
    "delta_vs_full": "Delta vs Full (Chênh lệch)",
    "epoch": "Epoch (Vòng huấn luyện)",
    "proposed_loss": "Validation Loss (Loss xác thực)",
    "proposed_score": "Primary Metric (Chỉ số chính)",
}


def display_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a display-only copy; pipeline keys and exported artifacts stay unchanged."""
    selected = frame if columns is None else frame[[col for col in columns if col in frame]]
    return selected.rename(columns=DISPLAY_LABELS)


def safe_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def metric_explanation(research: dict[str, Any]) -> str:
    metric = str((research.get("datasets_used") or [{}])[0].get("primary_metric", "task score"))
    normalized = metric.lower()
    if "loss" in normalized:
        meaning = "mất mát dự đoán; thấp hơn tốt hơn"
    elif any(term in normalized for term in ("accuracy", "f1", "bleu", "cider", "recall", "precision", "auc", "return", "success")):
        meaning = "điểm chất lượng nhiệm vụ; cao hơn tốt hơn"
    elif "wer" in normalized or "error" in normalized:
        meaning = "tỷ lệ lỗi; thấp hơn tốt hơn"
    else:
        meaning = "chỉ số chất lượng chính của nhiệm vụ"
    return f"{metric} — {meaning}"


def apply_preset() -> None:
    choice = st.session_state.get("preset", CUSTOM)
    if choice in PRESETS:
        for key, value in {"topic": choice, **PRESETS[choice]}.items():
            st.session_state[key] = value

