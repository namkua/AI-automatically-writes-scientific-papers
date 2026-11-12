from __future__ import annotations

import json
import os
import re
from typing import Any

from backend.research_engine import best_baseline_row, relative_metric_change
from backend.sanitizer import sanitize_academic_topic
from backend.schemas import PlannerOutput, ResearcherOutput, WriterOutput
from backend.synthesis_writer import MASTER_RESEARCH_WRITER_PROMPT, build_synthesis_writer

_EXHAUSTED_MODELS: set[str] = set()
_CANDIDATE_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)


def get_candidate_models(preferred: str | None = None) -> list[str]:
    base = list(_CANDIDATE_MODELS)
    if preferred:
        clean_pref = preferred.removeprefix("models/").strip()
        if clean_pref:
            return [clean_pref] + [m for m in base if m != clean_pref]
    return base


def _safe_repair_truncated_json(raw: str) -> dict[str, Any] | None:
    """Recover top-level string fields from a partially truncated JSON object."""
    result: dict[str, Any] = {}
    str_fields = (
        "title", "abstract", "introduction", "related_work",
        "methodology", "experimental_results", "discussion", "conclusion",
    )
    for field in str_fields:
        match = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)', raw, flags=re.DOTALL)
        if match:
            val = match.group(1)
            try:
                result[field] = json.loads(f'"{val}"', strict=False)
            except Exception:
                result[field] = val.replace(r"\n", "\n").replace(r'\"', '"')
    return result if len(result) >= 3 else None


def safe_json_loads(text: str) -> Any | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", cleaned)
        try:
            return json.loads(fixed, strict=False)
        except json.JSONDecodeError:
            return _safe_repair_truncated_json(cleaned)


def safe_live_call(
    api_key: str | None,
    prompt: str,
    max_tokens: int = 32768,
    model: str | None = None,
    response_schema: Any | None = None,
) -> Any | None:
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        timeout_ms = 45000 if max_tokens <= 8192 else 110000
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        preferred_model = (model or os.getenv("GEMINI_MODEL", "")).strip()
        candidates = get_candidate_models(preferred_model)
        active_candidates = [m for m in candidates if m not in _EXHAUSTED_MODELS] or candidates

        for model_name in active_candidates:
            try:
                print(f"[LiveGemini] Trying model={model_name} (max_tokens={max_tokens})...", flush=True)
                cfg_kwargs: dict[str, Any] = {
                    "temperature": 0.3,
                    "response_mime_type": "application/json",
                    "max_output_tokens": max_tokens,
                }
                if response_schema is not None:
                    cfg_kwargs["response_schema"] = response_schema
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    if hasattr(parsed, "model_dump"):
                        print(f"[LiveGemini] Success (parsed) with model={model_name}", flush=True)
                        return parsed.model_dump()
                    if isinstance(parsed, dict):
                        print(f"[LiveGemini] Success (parsed dict) with model={model_name}", flush=True)
                        return parsed
                text = getattr(response, "text", None)
                if text:
                    loaded = safe_json_loads(text)
                    if loaded is not None:
                        print(f"[LiveGemini] Success with model={model_name}, chars={len(text)}", flush=True)
                        return loaded
                    print(f"[LiveGemini] safe_json_loads returned None for model={model_name}", flush=True)
            except Exception as err:
                err_str = str(err)
                print(f"[LiveGemini] Model={model_name} error: {type(err).__name__}: {err}", flush=True)
                if any(code in err_str for code in ("NOT_FOUND", "404", "RESOURCE_EXHAUSTED", "429")):
                    _EXHAUSTED_MODELS.add(model_name)
                continue
        return None
    except Exception as err:
        print(f"[LiveGemini] Outer error: {type(err).__name__}: {err}", flush=True)
        return None


def _normalize_planner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    for ds in payload.get("datasets_catalog", []):
        if isinstance(ds, dict):
            ds["sample_size"] = str(ds.get("sample_size", "10,000 samples"))
    has_proposed = False
    for model_entry in payload.get("models_catalog", []):
        if isinstance(model_entry, dict):
            raw_role = str(model_entry.get("role", "")).strip().lower()
            if "propos" in raw_role:
                model_entry["role"] = "Proposed"
                has_proposed = True
            else:
                model_entry["role"] = "Baseline"
            for num_field, default_val in (("params_m", 125.0), ("flops_g", 15.0), ("latency_ms", 28.0), ("vram_mb", 512.0)):
                value = model_entry.get(num_field)
                if not isinstance(value, (int, float)):
                    match = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
                    model_entry[num_field] = float(match[0]) if match else default_val
                else:
                    model_entry[num_field] = float(value)
    if payload.get("models_catalog") and not has_proposed:
        payload["models_catalog"][-1]["role"] = "Proposed"
    for list_key in ("baseline_methods", "evaluation_metrics", "experimental_plan", "report_outline"):
        val = payload.get(list_key)
        if isinstance(val, str):
            payload[list_key] = [s.strip() for s in val.split("\n") if s.strip()]
        elif not isinstance(val, list) or not val:
            payload[list_key] = [f"Standard {list_key.replace('_', ' ')} protocol"]
        else:
            payload[list_key] = [str(x) for x in val]
    return payload


def _normalize_writer_payload(payload: dict[str, Any], plan: PlannerOutput) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if not payload.get("title"):
        payload["title"] = plan.title
    for str_key in ("abstract", "introduction", "related_work", "methodology", "experimental_results", "discussion", "conclusion"):
        val = payload.get(str_key)
        if isinstance(val, dict):
            payload[str_key] = "\n\n".join(f"### {k}\n{v}" for k, v in val.items())
        elif isinstance(val, list):
            payload[str_key] = "\n\n".join(str(x) for x in val)
        elif not val:
            payload[str_key] = f"Section {str_key} for {plan.title}."
        else:
            payload[str_key] = str(val)
    for list_key in ("keywords", "mathematical_formulas", "threats_to_validity", "references"):
        val = payload.get(list_key)
        if isinstance(val, str):
            payload[list_key] = [s.strip() for s in val.split("\n") if s.strip()]
        elif not isinstance(val, list) or not val:
            payload[list_key] = [f"Standard {list_key}"]
        else:
            payload[list_key] = [str(x) for x in val]
    return payload


def maybe_live_plan(
    topic: str,
    domain: str | None,
    objective: str | None,
    constraints: str | None,
    api_key: str | None,
    model: str | None = None,
) -> PlannerOutput | None:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        prompt = (
            "Return only valid JSON matching this schema: "
            "{title:str, domain:str, research_question:str, hypothesis:str, "
            "datasets_catalog:[{name:str, modality:str, sample_size:str, input_spec:str, primary_metric:str, description:str}], "
            "models_catalog:[{name:str, role:'Baseline'|'Proposed', backbone:str, params_m:float, flops_g:float, latency_ms:float, key_characteristic:str}], "
            "baseline_methods:[str], proposed_method:str, "
            "evaluation_metrics:[str], experimental_plan:[str], report_outline:[str]}. "
            "IMPORTANT: In models_catalog, include at least 2 items with role='Baseline' and 1 item with role='Proposed'. "
            "Generate one original proposed method, not a baseline-comparison paper. "
            "In models_catalog, role must be \"Baseline\" or \"Proposed\". params_m, flops_g, and latency_ms must be numeric floats. "
            f"Topic: {topic}; domain: {domain or 'AI research'}; objective: {objective or 'design a rigorous study'}; "
            f"constraints: {constraints or 'none specified'}. Use real, recognizable public datasets."
        )
        payload = safe_live_call(key, prompt, max_tokens=8192, model=model)
        if not isinstance(payload, dict):
            return None
        return PlannerOutput.model_validate(_normalize_planner_payload(payload))
    except Exception as err:
        print(f"[LiveGemini] Planner validation failed: {type(err).__name__}: {err}", flush=True)
        return None


def maybe_live_writer(
    plan: PlannerOutput,
    research: ResearcherOutput,
    api_key: str | None,
    model: str | None = None,
) -> WriterOutput | None:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    schema = "{title:str, abstract:str, keywords:[str], introduction:str, related_work:str, methodology:str, mathematical_formulas:[str], experimental_results:str, discussion:str, threats_to_validity:[str], conclusion:str, references:[str]}"
    prompt = (
        MASTER_RESEARCH_WRITER_PROMPT.format(sanitized_topic=sanitize_academic_topic(plan.title))
        + "\nReturn only valid JSON matching this exact schema (all section fields MUST be plain markdown strings, not nested objects): "
        + schema
        + ". Use the supplied benchmark rows for quantitative comparisons, preserving their evidence_type. "
        "Do not upgrade Literature Baseline or Controlled Simulation into Measured. "
        f"Topic: {plan.title}. Domain: {plan.domain}. Research question: {plan.research_question}. "
        f"Models: {json.dumps([item.model_dump() for item in plan.models_catalog])}. "
        f"Datasets: {json.dumps([item.model_dump() for item in research.datasets_used], ensure_ascii=False)}. "
        f"Benchmark rows (authoritative numeric evidence): {json.dumps([item.model_dump() for item in research.benchmark_table], ensure_ascii=False)}. "
        f"Training-curve summary: {json.dumps([item.model_dump() for item in research.training_curves if item.epoch in {1, 5, 10, 15}], ensure_ascii=False)}. "
        f"Hardware setup: {research.hardware_setup}. Key findings: {json.dumps(research.key_findings, ensure_ascii=False)}. "
        f"Knowledge Base: {json.dumps(plan.knowledge_base.model_dump())}. "
        f"Inference Engine: {json.dumps(plan.inference_engine.model_dump())}. "
        f"Multi-Agent Coordination: {json.dumps(plan.agent_coordination.model_dump())}. "
        f"Ablation Study: {json.dumps([item.model_dump() for item in research.ablation_study])}. "
        f"Evidence status: {research.evidence_mode}. Reasoning Consistency Score: {research.reasoning_consistency_score}. "
        f"Knowledge Retrieval Precision: {research.knowledge_retrieval_precision}."
    )
    try:
        payload = safe_live_call(key, prompt, max_tokens=32768, model=model, response_schema=WriterOutput)
        if not isinstance(payload, dict):
            return None
        fallback_dict = build_synthesis_writer(plan, research).model_dump()
        for k, default_val in fallback_dict.items():
            if not payload.get(k):
                payload[k] = default_val
        return WriterOutput.model_validate(_normalize_writer_payload(payload, plan))
    except Exception as err:
        print(f"[LiveGemini] Writer validation failed: {type(err).__name__}: {err}", flush=True)
        return None


def answer_paper_question(
    question: str,
    paper: WriterOutput,
    research: ResearcherOutput,
    api_key: str | None = None,
) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        try:
            qa_prompt = (
                "You are the Research Assistant for the following scientific manuscript. "
                "Answer the user's question accurately, concisely, and in the SAME language as the question (Vietnamese or English). "
                "Base your answer strictly on the manuscript and benchmark data below; note when figures are Controlled Planning Simulation Estimates. "
                'Return valid JSON: {"answer": "your response"}.\n\n'
                f"Title: {paper.title}\n"
                f"Abstract: {paper.abstract}\n"
                f"Methodology: {paper.methodology[:2000]}\n"
                f"Experimental Results: {paper.experimental_results[:2000]}\n"
                f"Benchmarks: {json.dumps([row.model_dump() for row in research.benchmark_table])}\n"
                f"Ablations: {json.dumps([row.model_dump() for row in research.ablation_study])}\n"
                f"Question: {question}"
            )
            payload = safe_live_call(key, qa_prompt)
            if isinstance(payload, dict) and isinstance(payload.get("answer"), str) and payload["answer"].strip():
                return payload["answer"].strip()
        except Exception:
            pass
    normalized = question.strip().lower()
    proposed = next((m for m in research.models_compared if m.role == "Proposed"), None)
    baselines = [m for m in research.models_compared if m.role == "Baseline"]
    proposed_row = next((row for row in research.benchmark_table if row.role == "Proposed"), None)
    baseline_rows = [row for row in research.benchmark_table if row.role == "Baseline"]
    primary_metric = research.datasets_used[0].primary_metric if research.datasets_used else "task score"
    strongest_baseline = best_baseline_row(baseline_rows, primary_metric)
    if any(term in normalized for term in ("baseline", "compare", "comparison", "so sánh")):
        names = ", ".join(m.name for m in baselines)
        strongest = strongest_baseline.model_name if strongest_baseline else "the strongest baseline"
        return f"The study compares {proposed.name if proposed else 'the proposed model'} against {names}. The strongest baseline in the simulated benchmark is {strongest}, while the proposed system is evaluated with the same dataset and reporting contract."
    if any(term in normalized for term in ("limitation", "weakness", "caveat", "hạn chế")):
        return "The main limitation is that the benchmark values, latency, and curves are deterministic planning simulations rather than measurements from instrumented training runs. A publication-ready study should add multiple seeds, confidence intervals, external validation, calibration, and hardware profiling."
    if any(term in normalized for term in ("dataset", "data", "benchmark", "dữ liệu")):
        catalog = ", ".join(d.name for d in research.datasets_used)
        return f"The generated evaluation uses three complementary datasets: {catalog}. They cover the core task, robustness under shift, and controlled ablations so the reported gain is not tied to a single split."
    if any(term in normalized for term in ("method", "model", "architecture", "mô hình", "phương pháp")):
        return f"The proposed method is {proposed.name if proposed else 'the task-aware model'}. It uses {proposed.backbone.lower() if proposed else 'adaptive signal selection'} to emphasize topic-relevant evidence while tracking quality, robustness, parameters, and latency together."
    if any(term in normalized for term in ("result", "score", "improve", "performance", "kết quả")) and proposed_row:
        baseline_score = strongest_baseline.primary_metric if strongest_baseline else 0.0
        gain = relative_metric_change(proposed_row.primary_metric, baseline_score, primary_metric)
        label = "relative improvement" if gain >= 0 else "relative degradation"
        return f"The proposed system reaches {proposed_row.primary_metric:.2f} on the primary metric versus {baseline_score:.2f} for the strongest baseline, a simulated {label} of {abs(gain):.1f}%. The result should be treated as a reproducible hypothesis until measured on real data."
    return f"This paper studies {paper.title}. Its central claim is that a topic-aware design can improve the target task while keeping robustness and deployment cost visible. Ask about the baseline comparison, datasets, method, results, or limitations for a more specific answer."

