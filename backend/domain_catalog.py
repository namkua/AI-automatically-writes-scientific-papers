from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.sanitizer import sanitize_academic_topic
from backend.schemas import (
    AgentCoordinationSpec,
    DatasetSpec,
    InferenceEngineSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    PlannerOutput,
    ProjectMetadata,
)

PTIT_COURSE_METADATA = {
    "institution": "Học viện Công nghệ Bưu chính Viễn thông (PTIT)",
    "faculty": "Khoa Công nghệ Thông tin",
    "subject": "Hệ thống thông minh",
    "instructor": "Thầy Hoàng Mạnh Cường",
    "student_name": "Trần Văn Đức",
    "student_id": "B23DCCN191",
}

_STOP_WORDS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "using", "via", "based", "towards", "study", "analysis", "system",
}

_DOMAIN_PROFILES = {
    "Speech & Audio Processing": {
        "keywords": ("speech", "audio", "whisper", "voice", "acoustic", "asr", "tts", "speaker recognition", "transcription"),
        "modality": "Audio / Speech",
        "metric": "WER / Speaker F1",
        "backbones": ("Conformer Baseline", "HuBERT", "Wav2Vec 2.0"),
        "proposed": "Dynamic Streaming Transducer (Ours)",
        "descriptor": "streaming acoustic representation",
    },
    "Reinforcement Learning & Robotics": {
        "keywords": ("reinforcement", "reinforcement learning", "rl", "q-learning", "q learning", "ppo", "sac", "policy", "robot", "robotics", "control", "agent", "navigation"),
        "modality": "State-action trajectories",
        "metric": "Episode return / Success rate",
        "backbones": ("PPO Baseline", "Soft Actor-Critic (SAC)", "Rainbow DQN"),
        "proposed": "Adaptive Contextual PPO (Ours)",
        "descriptor": "context-aware policy optimization",
    },
    "Multimodal & Vision-Language": {
        "keywords": ("multimodal", "multi-modal", "vision-language", "vision language", "clip", "vqa", "captioning", "diffusion", "text-to-image", "text to image", "image-text"),
        "modality": "Image-text pairs",
        "metric": "Recall@K / CIDEr",
        "backbones": ("CLIP Baseline", "BLIP-2", "Flamingo"),
        "proposed": "Cross-Modal Contrastive Fusion (Ours)",
        "descriptor": "cross-modal representation alignment",
    },
    "Recommendation Systems & CTR": {
        "keywords": ("recommendation", "recommender", "ctr", "collaborative filtering", "sequential recommendation", "ranking", "click-through", "personalization", "user-item"),
        "modality": "User-item interactions",
        "metric": "NDCG@10 / AUC",
        "backbones": ("DeepFM Baseline", "DLRM", "SASRec"),
        "proposed": "Dynamic Session-Aware Graph-Rec (Ours)",
        "descriptor": "session-aware ranking optimization",
    },
    "Graph Learning & FinTech": {
        "keywords": ("graph", "gnn", "gcn", "node", "edge", "financial", "finance", "fintech", "fraud", "transaction", "credit", "blockchain"),
        "modality": "Graph / Transaction records",
        "metric": "F1-score / PR-AUC",
        "backbones": ("GCN", "GraphSAGE", "Graph Attention Network (GAT)"),
        "proposed": "Adaptive Relational Graph Network (Ours)",
        "descriptor": "relational risk representation",
    },
    "Natural Language Processing (NLP) / RAG": {
        "keywords": ("retrieval", "retrieval-augmented", "retrieval augmented", "rag", "hallucination", "decoding", "attribution", "confidence-aware", "confidence aware", "llm", "large language model", "knowledge grounding", "source attribution"),
        "modality": "Text / Retrieved Evidence",
        "metric": "Faithfulness / EM",
        "backbones": ("DPR + Generator", "RAG-Sequence", "Self-RAG"),
        "proposed": "Confidence-Aware Attributed Decoder (Ours)",
        "descriptor": "retrieval-grounded confidence calibration",
    },
    "Natural Language Processing (NLP) / Machine Translation": {
        "keywords": ("translation", "machine translation", "cross-lingual", "cross lingual", "zero-shot", "low-resource", "low resource", "multilingual", "adapter", "mad-x", "mbert", "xlm-r", "flores", "tatoeba"),
        "modality": "Text / Translation",
        "metric": "COMET / BLEU",
        "backbones": ("mBERT Baseline", "XLM-RoBERTa Base", "MAD-X Adapters"),
        "proposed": "Dynamic Cross-Lingual Adapter (Ours)",
        "descriptor": "parameter-efficient cross-lingual adaptation",
    },
    "Quantum Computing": {
        "keywords": ("quantum", "qubit", "quantum circuit", "variational"),
        "modality": "Quantum circuit",
        "metric": "Circuit fidelity",
        "backbones": ("QAOA", "VQE", "Quantum kernel"),
        "proposed": "Adaptive Quantum Circuit",
        "descriptor": "noise-aware circuit optimization",
    },
    "Legal AI": {
        "keywords": ("legal", "law", "contract", "case", "statute", "compliance"),
        "modality": "Legal text",
        "metric": "Macro-F1",
        "backbones": ("Legal-BERT", "Longformer", "DeBERTa"),
        "proposed": "Neurosymbolic Legal Reasoner",
        "descriptor": "structured legal reasoning",
    },
    "Maritime Logistics": {
        "keywords": ("maritime", "shipping", "port", "vessel", "fleet", "logistics"),
        "modality": "Spatio-temporal tabular",
        "metric": "Route utility",
        "backbones": ("XGBoost", "LSTM forecaster", "Graph planner"),
        "proposed": "Adaptive Maritime Route Planner",
        "descriptor": "route-aware decision optimization",
    },
    "Computer Vision": {
        "keywords": ("image", "vision", "visual", "video", "segmentation", "detection", "medical imaging"),
        "modality": "Image",
        "metric": "mIoU",
        "backbones": ("ResNet", "ConvNeXt", "Vision Transformer"),
        "proposed": "Context-Aware Vision Model",
        "descriptor": "multi-scale visual representation",
    },
    "Natural Language Processing": {
        "keywords": ("language", "text", "nlp", "transformer", "sentiment", "document", "speech"),
        "modality": "Text",
        "metric": "Macro-F1",
        "backbones": ("BiLSTM", "BERT", "RoBERTa"),
        "proposed": "Context-Aware Language Model",
        "descriptor": "contextual representation learning",
    },
    "Graph Learning": {
        "keywords": ("graph", "gnn", "gcn", "node", "edge", "relationship", "molecule"),
        "modality": "Graph",
        "metric": "F1-score",
        "backbones": ("GCN", "GraphSAGE", "Graph Attention Network"),
        "proposed": "Adaptive Relational Graph Network",
        "descriptor": "relation-aware message passing",
    },
    "Reinforcement Learning": {
        "keywords": ("reinforcement", "robot", "robotics", "policy", "agent", "control", "navigation"),
        "modality": "State-action trajectory",
        "metric": "Episode return",
        "backbones": ("DQN", "PPO", "SAC"),
        "proposed": "Adaptive Policy Optimizer",
        "descriptor": "uncertainty-aware policy learning",
    },
    "Time Series Forecasting": {
        "keywords": ("forecast", "time series", "temporal", "stock", "demand", "weather", "energy"),
        "modality": "Time series",
        "metric": "Normalized RMSE",
        "backbones": ("ARIMA", "LSTM", "Temporal Transformer"),
        "proposed": "Adaptive Temporal Forecaster",
        "descriptor": "multi-scale temporal forecasting",
    },
    "Healthcare AI": {
        "keywords": ("health", "healthcare", "clinical", "patient", "diagnosis", "radiology", "medical"),
        "modality": "Clinical records",
        "metric": "AUROC",
        "backbones": ("Clinical MLP", "Survival model", "Medical Transformer"),
        "proposed": "Risk-Aware Clinical Model",
        "descriptor": "calibrated clinical prediction",
    },
    "FinTech": {
        "keywords": ("finance", "financial", "fintech", "fraud", "transaction", "credit", "market"),
        "modality": "Tabular and transaction graph",
        "metric": "PR-AUC",
        "backbones": ("XGBoost", "LightGBM", "GraphSAGE"),
        "proposed": "Risk-Aware Financial Model",
        "descriptor": "temporal risk representation",
    },
}

_DATASET_SOURCES = {
    "LibriSpeech ASR": "https://www.openslr.org/12",
    "VoxCeleb Speaker Recognition": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
    "Common Voice Multi-Lingual": "https://commonvoice.mozilla.org/datasets",
    "MS-COCO Captions": "https://cocodataset.org/#download",
    "VQA v2.0": "https://visualqa.org/download.html",
    "Conceptual Captions (CC3M)": "https://ai.google.com/research/ConceptualCaptions/",
    "MovieLens-20M": "https://grouplens.org/datasets/movielens/20m/",
    "Criteo Display Advertising": "https://ailab.criteo.com/ressources/",
    "Elliptic Bitcoin Dataset": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
    "Cora/Citeseer Graph": "https://github.com/kimiyoung/planetoid",
    "Natural Questions (NQ)": "https://ai.google.com/research/NaturalQuestions",
    "HotpotQA Multi-hop Attribution": "https://hotpotqa.github.io/",
    "TruthfulQA Faithfulness Set": "https://github.com/sylinrl/TruthfulQA",
    "Flores-101 (Low-Resource Languages)": "https://github.com/facebookresearch/flores",
    "XNLI Cross-Lingual Evaluation": "https://github.com/facebookresearch/XNLI",
    "Shakespeare Char": "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt",
    "Enwik8": "https://www.byronosa.com/enwik8",
    "Text8": "http://mattmahoney.net/dc/textdata.html",
}

_DOMAIN_TECH_STACKS = {
    "Natural Language Processing (NLP) / RAG": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "PEFT", "Accelerate", "FAISS"],
    "Natural Language Processing (NLP) / Machine Translation": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "PEFT", "Sacremoses", "Evaluate"],
    "Natural Language Processing": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "Tokenizers", "Accelerate"],
    "Computer Vision": ["PyTorch", "TorchVision", "timm", "Albumentations", "OpenCV", "TorchMetrics"],
    "Speech & Audio Processing": ["PyTorch", "Librosa", "torchaudio", "SpeechBrain", "Hugging Face Datasets", "JiWER"],
    "Reinforcement Learning & Robotics": ["PyTorch", "Gymnasium", "Stable-Baselines3", "Shimmy", "TensorBoard", "Farama Utils"],
    "Reinforcement Learning": ["PyTorch", "Gymnasium", "Stable-Baselines3", "TensorBoard"],
    "Graph Learning & FinTech": ["PyTorch", "PyTorch Geometric", "NetworkX", "OGB", "scikit-learn", "MLflow"],
    "Graph Learning": ["PyTorch", "PyTorch Geometric", "DGL", "NetworkX", "scikit-learn"],
    "Time Series Forecasting": ["PyTorch", "pandas", "scikit-learn", "StatsForecast", "sktime", "Darts"],
    "Multimodal & Vision-Language": ["PyTorch", "TorchVision", "Hugging Face Transformers", "OpenCLIP", "Albumentations", "W&B"],
}


def keyword_matches(text: str, keyword: str) -> bool:
    normalized = keyword.lower().strip()
    if len(normalized) <= 3 or normalized in {"asr", "tts", "rag", "llm", "clip", "vqa", "ctr", "ppo", "sac", "gpt"}:
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text) is not None
    return normalized in text


def topic_profile(topic: str, domain: str | None, objective: str | None = None) -> dict[str, Any]:
    clean_topic = sanitize_academic_topic(topic)
    text = f"{clean_topic} {domain or ''} {objective or ''}".lower()
    selected_name = next(
        (
            name
            for name, profile in _DOMAIN_PROFILES.items()
            if any(keyword_matches(text, kw) for kw in profile["keywords"])
        ),
        None,
    )
    if selected_name is None:
        selected_name = domain.strip() if domain and domain.strip() else "General Deep Learning & Applied AI"
        profile = {
            "keywords": (),
            "modality": "Domain-specific records",
            "metric": "Task performance",
            "backbones": ("Regularized MLP", "Gradient Boosting", "Attention Encoder"),
            "proposed": "Topic-Aware Adaptive Model",
            "descriptor": "task-aware signal selection",
        }
    else:
        profile = _DOMAIN_PROFILES[selected_name]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]+", clean_topic)
    keywords = list(dict.fromkeys(tok.lower() for tok in tokens if tok.lower() not in _STOP_WORDS))[:8]
    return {**profile, "topic": clean_topic, "domain": selected_name, "keywords": keywords}


def infer_domain(topic: str, domain: str | None, objective: str | None) -> str:
    return topic_profile(topic, domain, objective)["domain"]


def verified_local_data(domain: str, root: str | None = None) -> list[str]:
    data_root = Path(root or Path(__file__).resolve().parent.parent / "data")
    if "nlp" not in domain.lower() and domain != "Natural Language Processing":
        return []
    candidates = {
        "Shakespeare Char": data_root / "shakespeare_char" / "input.txt",
        "Enwik8": data_root / "enwik8" / "enwik8",
        "Text8": data_root / "text8" / "text8",
        "AG News": data_root / "huggingface" / "ag_news" / "train.jsonl",
        "IMDb Large Movie Review": data_root / "huggingface" / "imdb" / "train.jsonl",
    }
    return [name for name, path in candidates.items() if path.is_file() and path.stat().st_size > 0]


def validate_verified_request(topic: str, domain: str | None, objective: str | None) -> None:
    clean_topic = sanitize_academic_topic(topic)
    inferred = infer_domain(clean_topic, domain, objective)
    if not verified_local_data(inferred):
        raise HTTPException(
            status_code=422,
            detail=f"No verified local dataset is available for '{inferred}'. Disable verified-data checking to use the configured dataset catalog.",
        )


def knowledge_layer(
    topic: str, domain: str, objective: str | None, model_names: list[str], dataset_names: list[str]
) -> tuple[KnowledgeBaseSpec, InferenceEngineSpec, AgentCoordinationSpec]:
    profile = topic_profile(topic, domain, objective)
    text = f"{topic} {domain} {objective or ''}".lower()
    matched = [kw for kw in profile["keywords"] if keyword_matches(text, kw)]
    confidence = min(0.99, 0.62 + 0.06 * len(matched))
    kb = KnowledgeBaseSpec(
        taxonomy=[domain, profile["modality"], profile["metric"]],
        routing_rules=[
            "Prefer explicit domain input over inferred keywords.",
            "Use token-boundary matching for short technical acronyms.",
            "Route to complementary core, robustness, and ablation datasets.",
        ],
    )
    engine = InferenceEngineSpec(
        selected_domain=domain,
        matched_keywords=matched,
        routing_confidence=round(confidence, 2),
        routed_configuration=[
            f"Models: {', '.join(model_names)}",
            f"Datasets: {', '.join(dataset_names)}",
            f"Metric: {profile['metric']}",
        ],
    )
    coordination = AgentCoordinationSpec(
        agents=["Research Planner Agent", "Benchmark Generator Agent", "Metric Evaluator Agent", "Peer Reviewer Agent"],
        coordination_claim="Typed shared state lets each agent expose assumptions, evidence, metrics, and review feedback before manuscript export.",
    )
    return kb, engine, coordination


def ensure_system_intelligence(plan: PlannerOutput, topic: str, domain: str, objective: str | None) -> PlannerOutput:
    roles = {model.role for model in plan.models_catalog}
    if "Baseline" not in roles or "Proposed" not in roles:
        fallback_models = make_model_catalog(topic, domain)
        existing_names = {model.name for model in plan.models_catalog}
        plan.models_catalog = plan.models_catalog + [m for m in fallback_models if m.name not in existing_names]
        roles = {model.role for model in plan.models_catalog}
    if "Baseline" not in roles or "Proposed" not in roles:
        raise ValueError("Planner must provide at least one baseline and one proposed model.")
    plan.baseline_methods = [model.name for model in plan.models_catalog if model.role == "Baseline"]
    plan.proposed_method = next(model.name for model in plan.models_catalog if model.role == "Proposed")
    kb, engine, coordination = knowledge_layer(
        topic, domain, objective, [m.name for m in plan.models_catalog], [d.name for d in plan.datasets_catalog]
    )
    plan.project_metadata = ProjectMetadata(**PTIT_COURSE_METADATA)
    if not plan.knowledge_base.taxonomy:
        plan.knowledge_base = kb
    if not plan.inference_engine.selected_domain:
        plan.inference_engine = engine
    if not plan.agent_coordination.agents:
        plan.agent_coordination = coordination
    return plan


def domain_tech_stack(domain: str) -> list[str]:
    return _DOMAIN_TECH_STACKS.get(domain, ["Python", "PyTorch", "NumPy", "pandas", "scikit-learn", "MLflow"])


def dataset_source(name: str) -> str:
    return _DATASET_SOURCES.get(name, "https://huggingface.co/datasets")


def make_dataset_catalog(topic: str, domain: str | None) -> list[DatasetSpec]:
    profile = topic_profile(topic, domain)
    selected_domain = profile["domain"]
    if selected_domain == "Speech & Audio Processing":
        return [
            DatasetSpec(name="LibriSpeech ASR", modality="Speech / ASR", sample_size="1,000 hours of read English speech", input_spec="Audio waveforms with aligned transcripts", primary_metric="Word Error Rate", description="Standard automatic speech recognition benchmark for acoustic and language modeling."),
            DatasetSpec(name="VoxCeleb Speaker Recognition", modality="Speech / Speaker ID", sample_size="1.2M utterances from 7,000+ speakers", input_spec="Noisy speech segments with speaker identities", primary_metric="Speaker F1", description="In-the-wild speaker recognition benchmark with channel and accent variation."),
            DatasetSpec(name="Common Voice Multi-Lingual", modality="Speech / Multilingual", sample_size="100+ languages / community recordings", input_spec="Crowdsourced speech clips with language and transcript metadata", primary_metric="Macro WER", description="Multilingual speech benchmark for low-resource and accent-robust evaluation."),
        ]
    if selected_domain == "Reinforcement Learning & Robotics":
        return [
            DatasetSpec(name="OpenAI Gym / MuJoCo Suite", modality="Control / Simulation", sample_size="20 continuous-control environments", input_spec="State, action, reward, and next-state trajectories", primary_metric="Episode return", description="Continuous-control suite for reproducible policy optimization experiments."),
            DatasetSpec(name="DM Control Suite", modality="Control / Simulation", sample_size="30 physics-based tasks", input_spec="Pixel or state observations with dense rewards", primary_metric="Success rate", description="Physics-rich benchmark for robust control and generalization across tasks."),
            DatasetSpec(name="RoboSuite Manipulation", modality="Robot manipulation", sample_size="10 simulated manipulation tasks", input_spec="Robot proprioception, object state, and action trajectories", primary_metric="Task success", description="Manipulation benchmark for evaluating contextual policies under changing scenes."),
        ]
    if selected_domain == "Multimodal & Vision-Language":
        return [
            DatasetSpec(name="MS-COCO Captions", modality="Image-text pairs", sample_size="330k images / 1.5M captions", input_spec="Images paired with human-written descriptions", primary_metric="CIDEr / Recall@K", description="Image captioning and cross-modal retrieval benchmark."),
            DatasetSpec(name="VQA v2.0", modality="Image-question-answer", sample_size="200k images / 1.1M questions", input_spec="Images, natural-language questions, and answer distributions", primary_metric="VQA accuracy", description="Visual question answering benchmark designed to reduce language-only priors."),
            DatasetSpec(name="Conceptual Captions (CC3M)", modality="Web image-caption", sample_size="3.3M noisy image-caption pairs", input_spec="Weakly aligned web images and alt-text captions", primary_metric="Zero-shot Recall@K", description="Large-scale noisy pretraining and transfer benchmark for vision-language models."),
        ]
    if selected_domain == "Recommendation Systems & CTR":
        return [
            DatasetSpec(name="MovieLens-20M", modality="User-item ratings", sample_size="20M ratings / 138k users", input_spec="User, item, timestamp, and explicit rating interactions", primary_metric="NDCG@10", description="Collaborative filtering benchmark for personalized ranking."),
            DatasetSpec(name="Amazon Product Reviews", modality="User-item reviews", sample_size="Millions of reviews across product categories", input_spec="User, item, text review, rating, and temporal interactions", primary_metric="Recall@K", description="Long-tail recommendation benchmark with sparse sequential behavior."),
            DatasetSpec(name="Criteo Display Advertising", modality="Click-through logs", sample_size="45M anonymized ad impressions", input_spec="Categorical and numerical features with click labels", primary_metric="AUC / LogLoss", description="Large-scale CTR prediction benchmark under severe class imbalance."),
        ]
    if selected_domain == "Graph Learning & FinTech":
        return [
            DatasetSpec(name="Elliptic Bitcoin Dataset", modality="Transaction graph", sample_size="203k transactions / 234k edges", input_spec="Transaction nodes, temporal graph edges, and illicitness labels", primary_metric="PR-AUC", description="Bitcoin transaction graph for illicit activity and anomaly detection."),
            DatasetSpec(name="Cora/Citeseer Graph", modality="Citation graph", sample_size="10k papers / citation edges", input_spec="Node features, graph connectivity, and research labels", primary_metric="Node F1", description="Classic semi-supervised node-classification benchmark."),
            DatasetSpec(name="Financial Transaction Fraud Set", modality="Financial graph", sample_size="500k transactions / temporal holdout", input_spec="Accounts, transactions, and fraud risk labels", primary_metric="Fraud PR-AUC", description="Temporal fraud benchmark for relational risk modeling."),
        ]
    if selected_domain == "Natural Language Processing (NLP) / RAG":
        return [
            DatasetSpec(name="Natural Questions (NQ)", modality="Text / Open-domain QA", sample_size="307k questions / retrieved-document split", input_spec="Question, retrieved passages, and answer annotations", primary_metric="Exact Match / F1", description="Open-domain question answering benchmark for measuring retrieval-grounded generation quality."),
            DatasetSpec(name="HotpotQA Multi-hop Attribution", modality="Text / Multi-hop QA", sample_size="113k questions / supporting-fact annotations", input_spec="Questions with multi-document evidence chains", primary_metric="Joint F1 / Support EM", description="Multi-hop reasoning benchmark that evaluates whether generated claims are supported by sources."),
            DatasetSpec(name="TruthfulQA Faithfulness Set", modality="Text / Truthfulness", sample_size="817 adversarial questions / human references", input_spec="Adversarial prompts with truthful and hallucinated answer contrasts", primary_metric="Truthfulness / BLEU", description="Stress test for hallucination resistance, confidence calibration, and source attribution."),
        ]
    if selected_domain == "Natural Language Processing (NLP) / Machine Translation":
        return [
            DatasetSpec(name="Flores-101 (Low-Resource Languages)", modality="Text / Translation", sample_size="101 languages / train-validation-test splits", input_spec="Parallel sentences across high- and low-resource language pairs", primary_metric="COMET / BLEU", description="Multilingual translation benchmark for evaluating zero-shot and low-resource transfer."),
            DatasetSpec(name="XNLI Cross-Lingual Evaluation", modality="Text / NLI", sample_size="15 languages / matched validation sets", input_spec="Premise-hypothesis pairs translated across languages", primary_metric="Accuracy", description="Cross-lingual natural language inference suite for transfer and representation quality."),
            DatasetSpec(name="Tatoeba Translation Benchmark", modality="Text / Parallel Sentences", sample_size="50,000 sentence pairs / held-out evaluation", input_spec="Multilingual parallel sentence pairs with language-direction labels", primary_metric="BLEU / chrF", description="Translation benchmark for measuring robustness across diverse language directions."),
        ]
    if selected_domain == "Natural Language Processing":
        topic_text = topic.lower()
        if any(term in topic_text for term in ("classification", "sentiment", "news", "text classifier")):
            return [
                DatasetSpec(name="AG News", modality="Text classification", sample_size="120k labeled news articles", input_spec="Article text with four topic labels", primary_metric="Accuracy / Macro-F1", description="Real-world news topic classification benchmark suitable for supervised transformer experiments."),
                DatasetSpec(name="IMDb Large Movie Review", modality="Sentiment text", sample_size="50k labeled movie reviews", input_spec="Review text with binary sentiment labels", primary_metric="Accuracy / Macro-F1", description="Balanced sentiment benchmark for robust text classification and error analysis."),
                DatasetSpec(name="Shakespeare Char", modality="Character-level text", sample_size="1.1 MB corpus / train-validation split", input_spec="Encoded character sequences with next-character targets", primary_metric="Validation loss", description="Local character-level corpus for a complementary language-modeling check."),
            ]
        return [
            DatasetSpec(name="Shakespeare Char", modality="Character-level text", sample_size="1.1 MB corpus / train-validation split", input_spec="Encoded character sequences with next-character targets", primary_metric="Validation loss", description="Shakespeare play text used for character-level language modeling and qualitative generation."),
            DatasetSpec(name="Enwik8", modality="Character-level text", sample_size="100 MB Wikipedia corpus / train-validation split", input_spec="Raw Wikipedia characters stored as uint16 token ids", primary_metric="Validation loss", description="Standard 100 MB enwik8 compression and next-character prediction benchmark."),
            DatasetSpec(name="Text8", modality="Character-level text", sample_size="100 MB normalized English corpus / train-validation split", input_spec="Normalized lowercase English characters without formatting", primary_metric="Validation loss", description="Clean text8 corpus for measuring language-model convergence on standardized text."),
        ]
    anchor = " ".join(word.title() for word in profile["keywords"][:3]) or selected_domain
    return [
        DatasetSpec(name=f"{anchor} Core Benchmark", modality=profile["modality"], sample_size="40k train / 8k validation / 8k test", input_spec=f"Topic-specific {profile['modality'].lower()} with supervised targets", primary_metric=profile["metric"], description=f"Primary simulated benchmark for {profile['topic']} with realistic distribution shift."),
        DatasetSpec(name=f"{anchor} Robustness Split", modality=profile["modality"], sample_size="18k samples / temporal or OOD holdout", input_spec="Stress-test partition with noise, missingness, and edge cases", primary_metric=profile["metric"], description="Challenge split for measuring generalization beyond the nominal data distribution."),
        DatasetSpec(name=f"{anchor} Ablation Set", modality=profile["modality"], sample_size="12k samples / controlled factors", input_spec="Matched subset for component removal and interpretability checks", primary_metric="Robustness gap", description="Controlled evaluation set for isolating the contribution of each model component."),
    ]


def make_model_catalog(topic: str, domain: str | None) -> list[ModelSpec]:
    profile = topic_profile(topic, domain)
    selected_domain = profile["domain"]
    if selected_domain == "Speech & Audio Processing":
        return [
            ModelSpec(name="Conformer Baseline", role="Baseline", backbone="Conformer encoder", params_m=118.0, flops_g=42.0, latency_ms=35.0, key_characteristic="Convolution-augmented transformer baseline for speech recognition."),
            ModelSpec(name="HuBERT", role="Baseline", backbone="Self-supervised speech encoder", params_m=95.0, flops_g=38.0, latency_ms=41.0, key_characteristic="Masked prediction pretraining for robust acoustic representations."),
            ModelSpec(name="Wav2Vec 2.0", role="Baseline", backbone="Self-supervised waveform encoder", params_m=84.0, flops_g=34.0, latency_ms=39.0, key_characteristic="Learns speech representations directly from raw waveforms."),
            ModelSpec(name="Dynamic Streaming Transducer (Ours)", role="Proposed", backbone="Streaming transducer with adaptive context", params_m=72.0, flops_g=28.0, latency_ms=24.0, key_characteristic="Adaptive streaming context for low-latency multilingual transcription."),
        ]
    if selected_domain == "Reinforcement Learning & Robotics":
        return [
            ModelSpec(name="PPO Baseline", role="Baseline", backbone="Proximal Policy Optimization", params_m=4.2, flops_g=1.2, latency_ms=8.0, key_characteristic="Stable clipped policy-gradient baseline for continuous control."),
            ModelSpec(name="Soft Actor-Critic (SAC)", role="Baseline", backbone="Maximum-entropy actor-critic", params_m=5.1, flops_g=1.5, latency_ms=9.0, key_characteristic="Entropy-regularized off-policy control baseline."),
            ModelSpec(name="Rainbow DQN", role="Baseline", backbone="Distributional value-based agent", params_m=8.4, flops_g=2.8, latency_ms=12.0, key_characteristic="Combines value-learning improvements for discrete action tasks."),
            ModelSpec(name="Adaptive Contextual PPO (Ours)", role="Proposed", backbone="PPO with contextual policy gating", params_m=5.8, flops_g=1.7, latency_ms=9.5, key_characteristic="Adapts policy context to changing dynamics and observation regimes."),
        ]
    if selected_domain == "Multimodal & Vision-Language":
        return [
            ModelSpec(name="CLIP Baseline", role="Baseline", backbone="Dual-encoder contrastive model", params_m=151.0, flops_g=68.0, latency_ms=42.0, key_characteristic="Strong zero-shot image-text alignment baseline."),
            ModelSpec(name="BLIP-2", role="Baseline", backbone="Frozen vision encoder + Q-Former", params_m=188.0, flops_g=82.0, latency_ms=56.0, key_characteristic="Bridges frozen vision and language models through query tokens."),
            ModelSpec(name="Flamingo", role="Baseline", backbone="Few-shot vision-language decoder", params_m=340.0, flops_g=120.0, latency_ms=74.0, key_characteristic="Interleaves visual context with language generation."),
            ModelSpec(name="Cross-Modal Contrastive Fusion (Ours)", role="Proposed", backbone="Gated image-text fusion encoder", params_m=172.0, flops_g=76.0, latency_ms=48.0, key_characteristic="Jointly calibrates visual and textual evidence under a contrastive objective."),
        ]
    if selected_domain == "Recommendation Systems & CTR":
        return [
            ModelSpec(name="DeepFM Baseline", role="Baseline", backbone="Factorization machine + deep network", params_m=18.0, flops_g=4.5, latency_ms=15.0, key_characteristic="Models low- and high-order feature interactions for CTR."),
            ModelSpec(name="DLRM", role="Baseline", backbone="Embedding interaction recommender", params_m=32.0, flops_g=6.2, latency_ms=18.0, key_characteristic="Production-oriented dense and sparse feature interaction model."),
            ModelSpec(name="SASRec", role="Baseline", backbone="Self-attentive sequential recommender", params_m=24.0, flops_g=5.4, latency_ms=21.0, key_characteristic="Captures sequential user intent with causal self-attention."),
            ModelSpec(name="Dynamic Session-Aware Graph-Rec (Ours)", role="Proposed", backbone="Session graph with adaptive ranking", params_m=29.0, flops_g=6.0, latency_ms=19.0, key_characteristic="Combines session context and relational item transitions for ranking."),
        ]
    if selected_domain == "Graph Learning & FinTech":
        return [
            ModelSpec(name="GCN", role="Baseline", backbone="Graph convolutional network", params_m=12.0, flops_g=2.8, latency_ms=18.0, key_characteristic="Spectral-style neighborhood aggregation baseline."),
            ModelSpec(name="GraphSAGE", role="Baseline", backbone="Inductive neighbor sampler", params_m=16.0, flops_g=3.4, latency_ms=21.0, key_characteristic="Scalable inductive representation learning for unseen nodes."),
            ModelSpec(name="Graph Attention Network (GAT)", role="Baseline", backbone="Attention-based message passing", params_m=19.0, flops_g=4.1, latency_ms=25.0, key_characteristic="Learns neighbor importance through attention coefficients."),
            ModelSpec(name="Adaptive Relational Graph Network (Ours)", role="Proposed", backbone="Dynamic relation-aware GNN", params_m=22.0, flops_g=4.5, latency_ms=24.0, key_characteristic="Adapts relation weights for temporal and heterogeneous graph risk patterns."),
        ]
    if selected_domain == "Natural Language Processing (NLP) / RAG":
        return [
            ModelSpec(name="DPR + Generator", role="Baseline", backbone="Dense Passage Retriever + seq2seq generator", params_m=410.0, flops_g=82.0, latency_ms=68.0, key_characteristic="Two-stage retrieval-generation baseline with no explicit confidence control."),
            ModelSpec(name="RAG-Sequence", role="Baseline", backbone="Retriever-conditioned sequence generator", params_m=430.0, flops_g=88.0, latency_ms=72.0, key_characteristic="Marginalizes retrieved passages during generation but does not expose claim-level evidence."),
            ModelSpec(name="Self-RAG", role="Baseline", backbone="Reflection-token augmented LLM", params_m=750.0, flops_g=145.0, latency_ms=96.0, key_characteristic="Uses self-reflection signals to decide when retrieval and critique are needed."),
            ModelSpec(name="Confidence-Aware Attributed Decoder (Ours)", role="Proposed", backbone="RAG decoder with confidence gating and source attribution", params_m=465.0, flops_g=91.0, latency_ms=61.0, key_characteristic="Calibrates decoding confidence and attaches generated claims to retrieved evidence to reduce unsupported statements."),
        ]
    if selected_domain == "Natural Language Processing (NLP) / Machine Translation":
        return [
            ModelSpec(name="mBERT Baseline", role="Baseline", backbone="BERT-Base-Multilingual", params_m=177.0, flops_g=35.0, latency_ms=38.2, key_characteristic="Multilingual encoder baseline for zero-shot transfer across language families."),
            ModelSpec(name="XLM-RoBERTa Base", role="Baseline", backbone="XLM-R", params_m=270.0, flops_g=52.0, latency_ms=45.1, key_characteristic="Large multilingual masked-language encoder with broad cross-lingual coverage."),
            ModelSpec(name="MAD-X Adapters", role="Baseline", backbone="XLM-R + Adapter", params_m=28.4, flops_g=14.0, latency_ms=22.5, key_characteristic="Parameter-efficient adapter composition for multilingual transfer."),
            ModelSpec(name="Dynamic Cross-Lingual Adapter (Ours)", role="Proposed", backbone="XLM-R + Dynamic Gating", params_m=14.2, flops_g=10.5, latency_ms=18.6, key_characteristic="Dynamically gates language and task adapters to improve low-resource transfer with a compact trainable budget."),
        ]
    if selected_domain == "Natural Language Processing":
        return [
            ModelSpec(name="nanoGPT Baseline", role="Baseline", backbone="Decoder-only GPT Transformer", params_m=12.4, flops_g=6.8, latency_ms=24.0, key_characteristic="Reference character-level nanoGPT with causal self-attention and next-character prediction."),
            ModelSpec(name="nanoGPT + QK-Norm", role="Baseline", backbone="GPT Transformer with normalized Q/K", params_m=12.4, flops_g=7.0, latency_ms=25.0, key_characteristic="Attention-stability variant that normalizes query and key representations."),
            ModelSpec(name="nanoGPT + Layer-wise LR", role="Baseline", backbone="GPT Transformer with layer-wise learning rates", params_m=12.4, flops_g=6.8, latency_ms=24.5, key_characteristic="Optimization baseline assigning depth-aware learning-rate schedules."),
            ModelSpec(name="Dynamic Attention Head Gating nanoGPT", role="Proposed", backbone="GPT Transformer with learnable head gates", params_m=12.5, flops_g=7.1, latency_ms=25.5, key_characteristic="Proposed nanoGPT variant with sigmoid head gates and sparsity-aware attention allocation."),
        ]
    if selected_domain == "General Deep Learning & Applied AI":
        return [
            ModelSpec(name="ResNet-50", role="Baseline", backbone="Residual convolutional encoder", params_m=25.6, flops_g=4.1, latency_ms=32.0, key_characteristic="Strong general-purpose residual baseline for supervised representation learning."),
            ModelSpec(name="Transformer Base", role="Baseline", backbone="Encoder-decoder transformer", params_m=65.0, flops_g=12.0, latency_ms=44.0, key_characteristic="Neutral attention-based baseline for sequence and structured prediction tasks."),
            ModelSpec(name="MLP Baseline", role="Baseline", backbone="Regularized multilayer perceptron", params_m=8.2, flops_g=1.2, latency_ms=18.0, key_characteristic="Compact non-relational baseline for measuring the value of architectural structure."),
            ModelSpec(name="Task-Adaptive Deep Network (Ours)", role="Proposed", backbone="Adaptive feature encoder", params_m=19.5, flops_g=3.8, latency_ms=27.0, key_characteristic="Task-adaptive fusion model used as a neutral proposed method for unknown topics."),
        ]
    models = [
        ModelSpec(
            name=backbone,
            role="Baseline",
            backbone=f"{backbone} architecture",
            params_m=round(18.0 + idx * 13.0 + len(profile["keywords"]) * 0.7, 1),
            flops_g=round(4.5 + idx * 3.2, 1),
            latency_ms=round(24.0 + idx * 11.0, 1),
            key_characteristic=f"Established {selected_domain} baseline with a distinct capacity and inductive bias.",
        )
        for idx, backbone in enumerate(profile["backbones"])
    ]
    models.append(
        ModelSpec(
            name=profile["proposed"],
            role="Proposed",
            backbone=f"Hybrid {profile['descriptor']} architecture",
            params_m=round(30.0 + len(profile["keywords"]) * 1.4, 1),
            flops_g=round(8.0 + len(profile["keywords"]) * 0.8, 1),
            latency_ms=round(31.0 + len(profile["keywords"]) * 1.2, 1),
            key_characteristic=f"Combines {profile['descriptor']} with adaptive regularization and uncertainty-aware evaluation.",
        )
    )
    return models

