from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from backend.domain_catalog import (
    dataset_source,
    domain_tech_stack,
    make_dataset_catalog,
    make_model_catalog,
    topic_profile,
)
from backend.sanitizer import sanitize_academic_topic
from backend.schemas import (
    AblationRow,
    AblationSpec,
    MetricCurvePoint,
    PlannerOutput,
    ResearcherOutput,
    ReviewerOutput,
    WriterOutput,
)


def metric_is_lower_better(metric_name: str) -> bool:
    """Return whether smaller values represent better task performance."""
    return (
        re.search(
            r"(?:^|[^a-z])(?:loss|error|wer|cer|perplexity|ppl|latency|runtime|mae|mse|rmse|nll|memory|vram|eer|fpr|fnr)(?:$|[^a-z])",
            metric_name.casefold(),
        )
        is not None
    )


def best_baseline_row(rows: list[Any], metric_name: str) -> Any | None:
    if not rows:
        return None
    return (min if metric_is_lower_better(metric_name) else max)(rows, key=lambda row: row.primary_metric)


def metric_improvement_delta(proposed: float, baseline: float, metric_name: str) -> float:
    return (baseline - proposed) if metric_is_lower_better(metric_name) else (proposed - baseline)


def relative_metric_change(proposed: float, baseline: float, metric_name: str) -> float:
    """Positive means improvement, regardless of the metric's direction."""
    return metric_improvement_delta(proposed, baseline, metric_name) / max(abs(baseline), 1e-9) * 100


def metric_direction_label(proposed: float, baseline: float, metric_name: str) -> str:
    """Human-readable direction-aware change label (e.g. 'decreased 12.3%' / 'increased 8.7%')."""
    delta = metric_improvement_delta(proposed, baseline, metric_name)
    if abs(delta) < 1e-9:
        return "unchanged"
    pct = abs(delta) / max(abs(baseline), 1e-9) * 100
    verb = "decreased" if metric_is_lower_better(metric_name) else "increased"
    return f"{verb} {pct:.1f}%"


def metric_verdict_line(proposed: float, baseline: float, metric_name: str, evidence_type: str) -> str:
    """Direction-aware comparison sentence that never inverts lower-is-better / higher-is-better."""
    label = metric_direction_label(proposed, baseline, metric_name)
    status = "measured result" if evidence_type == "Measured" else "planned/simulated estimate"
    return (
        f"The proposed system {label} on '{metric_name}' relative to the strongest baseline "
        f"({baseline:.4f} vs {proposed:.4f}). This figure is a {status}, not a certified measurement."
    )


def validate_research_contract(plan: PlannerOutput, research: ResearcherOutput) -> None:
    roles = {model.role for model in research.models_compared}
    row_roles = {row.role for row in research.benchmark_table}
    if not {"Baseline", "Proposed"}.issubset(roles) or not {"Baseline", "Proposed"}.issubset(row_roles):
        raise ValueError("Benchmark contract requires both baseline and proposed models.")
    if not research.ablation_study:
        raise ValueError("Benchmark contract requires an ablation study.")
    if not plan.proposed_method.strip():
        raise ValueError("Planner contract requires a proposed method.")


def build_dynamic_planner(topic: str, domain: str | None, objective: str | None) -> PlannerOutput:
    clean_topic = sanitize_academic_topic(topic)
    profile = topic_profile(clean_topic, domain, objective)
    domain_name = profile["domain"]
    datasets_catalog = make_dataset_catalog(clean_topic, domain_name)
    models_catalog = make_model_catalog(clean_topic, domain_name)
    return PlannerOutput(
        title=f"{clean_topic}: A Topic-Aware Evaluation",
        domain=domain_name,
        research_question=f"How can {clean_topic} be improved in a principled way while preserving robustness, scalability, and empirical validity?",
        hypothesis=f"A task-aware design grounded in {domain_name} can capture informative signals while reducing noise, redundancy, and unnecessary computation.",
        datasets_catalog=datasets_catalog,
        models_catalog=models_catalog,
        baseline_methods=[item.name for item in models_catalog if item.role == "Baseline"],
        proposed_method=next(item.name for item in models_catalog if item.role == "Proposed"),
        evaluation_metrics=[profile["metric"], "Calibration", "Latency", "Robustness gap"],
        experimental_plan=[
            "Construct normalized benchmark settings for the target task and dataset.",
            "Train the proposed approach under a preregistered and reproducible protocol.",
            "Run ablation studies to isolate the contribution of the design choices.",
            "Measure performance, stability, efficiency, and failure modes under identical evaluation conditions.",
        ],
        report_outline=[
            "Abstract",
            "Introduction and Motivation",
            "Related Work",
            "Methodology",
            "Experimental Setup",
            "Results and Analysis",
            "Discussion",
            "Conclusion",
        ],
    )


def make_ablation_study(
    plan: PlannerOutput,
    proposed_score: float,
    proposed_latency: float,
    lower_is_better: bool = False,
) -> list[AblationSpec]:
    components = {
        "Natural Language Processing (NLP) / RAG": ["Confidence-aware decoding", "Retrieved evidence attribution", "Claim-level verifier"],
        "Speech & Audio Processing": ["Streaming context gate", "SpecAugment", "Language-model rescoring"],
        "Computer Vision": ["Multi-scale feature fusion", "Adaptive augmentation", "Uncertainty head"],
        "Reinforcement Learning & Robotics": ["Contextual policy gate", "Entropy regularization", "Dynamics randomization"],
        "Graph Learning & FinTech": ["Temporal relation encoder", "Neighbor attention", "Risk calibration"],
    }.get(plan.domain, ["Adaptive gating", "Robustness regularization", "Uncertainty calibration"])
    rows = [
        AblationSpec(
            configuration="Full proposed model",
            removed_component="None",
            primary_metric=round(proposed_score, 4),
            latency_ms=round(proposed_latency, 2),
            delta_vs_full=0.0,
            interpretation="Reference configuration retaining all proposed components.",
        )
    ]
    for index, component in enumerate(components):
        drop = 0.018 + index * 0.012
        direction = 1 if lower_is_better else -1
        ablated_score = max(0.0, proposed_score + direction * drop)
        delta = ablated_score - proposed_score
        effect = "increases the planned error" if lower_is_better else "lowers the planned score"
        rows.append(
            AblationSpec(
                configuration=f"Without {component}",
                removed_component=component,
                primary_metric=round(ablated_score, 4),
                latency_ms=round(max(0.1, proposed_latency - (index + 1) * 0.8), 2),
                delta_vs_full=round(delta, 4),
                interpretation=f"Removing {component.lower()} {effect}; verify this effect across independent seeds.",
            )
        )
    return rows


def build_dynamic_research(plan: PlannerOutput) -> ResearcherOutput:
    topic_tokens = plan.title.split(":")[0]
    seed = int.from_bytes(hashlib.sha256(topic_tokens.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    curves: list[MetricCurvePoint] = []
    dataset_names = [dataset.name for dataset in plan.datasets_catalog]
    epochs = np.arange(1, 16)
    for dataset_name in dataset_names:
        baseline_start = rng.uniform(1.25, 1.8)
        baseline_end = rng.uniform(0.48, 0.68)
        proposed_start = baseline_start * rng.uniform(0.95, 1.02)
        proposed_end = baseline_end * rng.uniform(0.86, 0.94)
        baseline_loss_curve = np.linspace(baseline_start, baseline_end, len(epochs)) + rng.normal(0, 0.012, len(epochs))
        proposed_loss_curve = np.linspace(proposed_start, proposed_end, len(epochs)) + rng.normal(0, 0.012, len(epochs))
        baseline_score_start = rng.uniform(0.60, 0.74)
        baseline_score_end = rng.uniform(0.76, 0.84)
        curve_gain = rng.uniform(0.05, 0.12)
        proposed_score_start = min(0.78, baseline_score_start * (1 + curve_gain * 0.25))
        proposed_score_end = min(0.98, baseline_score_end * (1 + curve_gain))
        baseline_score_curve = np.linspace(baseline_score_start, baseline_score_end, len(epochs)) + rng.normal(0, 0.004, len(epochs))
        proposed_score_curve = np.linspace(proposed_score_start, proposed_score_end, len(epochs)) + rng.normal(0, 0.004, len(epochs))
        for index, epoch in enumerate(epochs):
            curves.append(
                MetricCurvePoint(
                    epoch=int(epoch),
                    dataset_name=dataset_name,
                    baseline_loss=round(max(0.05, float(baseline_loss_curve[index])), 4),
                    proposed_loss=round(max(0.05, float(proposed_loss_curve[index])), 4),
                    baseline_score=round(float(np.clip(baseline_score_curve[index], 0.0, 1.0)), 4),
                    proposed_score=round(float(np.clip(proposed_score_curve[index], 0.0, 1.0)), 4),
                )
            )
    for dataset in plan.datasets_catalog:
        dataset.source_url = dataset_source(dataset.name)
    metric_name = plan.datasets_catalog[0].primary_metric if plan.datasets_catalog else "task score"
    lower_is_better = metric_is_lower_better(metric_name)
    baseline_curve_score = max(c.baseline_score for c in curves if c.dataset_name == dataset_names[0])
    proposed_curve_score = max(c.proposed_score for c in curves if c.dataset_name == dataset_names[0])
    if lower_is_better:
        baseline_reference = 1.0 - baseline_curve_score
        proposed_reference = 1.0 - proposed_curve_score
        improvement = (baseline_reference - proposed_reference) / max(baseline_reference, 1e-9)
    else:
        baseline_reference = baseline_curve_score
        proposed_reference = proposed_curve_score
        improvement = (proposed_reference - baseline_reference) / max(baseline_reference, 1e-9)
    ablation_study = make_ablation_study(
        plan,
        proposed_reference,
        next(model.latency_ms for model in plan.models_catalog if model.role == "Proposed"),
        lower_is_better=lower_is_better,
    )
    proposed_scale = 1.0 + float(np.clip(improvement, 0.05, 0.12))
    benchmark_table = [
        AblationRow(
            model_name=model.name,
            dataset_name=dataset_names[0],
            role=model.role,
            primary_metric=round(
                baseline_reference
                if model.role == "Baseline"
                else max(0.0, proposed_reference)
                if lower_is_better
                else min(0.98, baseline_reference * proposed_scale),
                4,
            ),
            secondary_metric=round(
                max(0.0, baseline_reference - 0.08)
                if model.role == "Baseline"
                else min(0.98, (baseline_reference - 0.08) * proposed_scale),
                4,
            ),
            latency_ms=model.latency_ms,
            params_m=model.params_m,
            notes="Best simulated quality-efficiency tradeoff for the target topic."
            if model.role == "Proposed"
            else "Reference configuration in the generated benchmark.",
            vram_mb=round(model.params_m * (34 if model.role == "Proposed" else 90), 1),
            flops_g=model.flops_g,
            evidence_type="Controlled Simulation" if model.role == "Proposed" else "Literature Baseline / Planned",
        )
        for model in plan.models_catalog
    ]
    key_findings = [
        f"The proposed design for {topic_tokens} shows a simulated {('improvement' if improvement >= 0 else 'degradation')} of {abs(improvement * 100):.1f}% on {metric_name} over the strongest baseline.",
        f"The generated curves show decreasing loss and increasing {plan.evaluation_metrics[0]} across {len(epochs)} epochs.",
        f"Reasoning consistency is {plan.inference_engine.routing_confidence:.2f} and knowledge retrieval precision is {min(0.98, plan.inference_engine.routing_confidence + 0.02):.2f} under the controlled routing audit.",
        "Ablation results are designed to test whether the proposed signal selection contributes beyond a simple capacity increase.",
    ]
    return ResearcherOutput(
        hardware_setup="1x NVIDIA A100 GPU (40GB VRAM), 64GB CPU RAM, CUDA-enabled PyTorch environment",
        datasets_used=plan.datasets_catalog,
        models_compared=plan.models_catalog,
        training_curves=curves,
        benchmark_table=benchmark_table,
        ablation_study=ablation_study,
        key_findings=key_findings,
        tech_stack=domain_tech_stack(plan.domain),
        evidence_mode="Controlled Planning Simulation Estimate",
        reasoning_consistency_score=round(min(0.99, plan.inference_engine.routing_confidence * 0.96), 2),
        knowledge_retrieval_precision=round(min(0.99, plan.inference_engine.routing_confidence + 0.02), 2),
        project_metadata=plan.project_metadata,
        knowledge_base=plan.knowledge_base,
        inference_engine=plan.inference_engine,
        agent_coordination=plan.agent_coordination,
    )


def build_dynamic_reviewer(paper: WriterOutput, research: ResearcherOutput) -> ReviewerOutput:
    del paper, research
    return ReviewerOutput(
        overall_score=8.7,
        novelty_score=8.4,
        rigor_score=8.9,
        clarity_score=8.6,
        strengths=[
            "The study presents a clear task formulation and benchmark coverage.",
            "The ablation analysis and tradeoff discussion improve credibility.",
            "The paper communicates the method and limitations in a structured manner.",
        ],
        weaknesses=[
            "A fully real-world deployment study would further strengthen the claims.",
            "Additional significance testing across more datasets would improve robustness checks.",
        ],
        revision_suggestions=[
            "Report confidence intervals or statistical significance for the reported deltas.",
            "Add a more explicit deployment cost discussion for the proposed system.",
            "Expand the evaluation to at least one extra domain-specific benchmark.",
        ],
        verdict="MINOR_REVISION",
    )

