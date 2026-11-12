from __future__ import annotations

import hmac
import ipaddress
import os
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

from backend.domain_catalog import (  # noqa: E402
    PTIT_COURSE_METADATA,
    ensure_system_intelligence,
    infer_domain,
    validate_verified_request,
)
from backend.gemini_client import (  # noqa: E402
    _EXHAUSTED_MODELS,
    answer_paper_question as _answer_paper_question,
    maybe_live_plan,
    maybe_live_writer,
    safe_live_call as _safe_live_call,
)
from backend.guardrail_agent import audit_manuscript  # noqa: E402
from backend.research_engine import (  # noqa: E402
    build_dynamic_planner,
    build_dynamic_research,
    build_dynamic_reviewer,
    validate_research_contract,
)
from backend.sanitizer import sanitize_academic_topic, sanitize_writer_output  # noqa: E402
from backend.schemas import (  # noqa: E402
    FullPipelineResponse,
    GuardrailOutput,
    PipelineRequest,
    PlannerOutput,
    ResearcherOutput,
    ReviewerOutput,
    TopicInput,
    WriterOutput,
)
from backend.synthesis_writer import (  # noqa: E402
    academic_abstract,
    build_synthesis_writer,
    ensure_gap_and_proposal_sections,
)

app = FastAPI(
    title="Smart Research Lab API",
    version="0.2.0",
    description="Multi-agent research pipeline with live Gemini integration and topic-aware fallback generation.",
)


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOWED_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins:
        origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
    if "*" in origins:
        raise RuntimeError("CORS_ALLOWED_ORIGINS must list explicit origins; wildcard origins are disabled.")
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-API-Key"],
)

try:
    _RATE_LIMIT_PER_MINUTE = max(1, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "10")))
except ValueError:
    _RATE_LIMIT_PER_MINUTE = 10
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


@app.middleware("http")
async def protect_api(request: Request, call_next):
    """Require X-API-Key for remote requests and enforce a per-client in-memory rate limit."""
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)

    client_host = request.client.host if request.client else "unknown"
    expected_key = os.getenv("SMART_RESEARCH_API_KEY", "").strip()
    if expected_key:
        supplied_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied_key.encode("utf-8"), expected_key.encode("utf-8")):
            return JSONResponse(status_code=401, content={"detail": "A valid X-API-Key header is required."})
    else:
        try:
            is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            return JSONResponse(
                status_code=403,
                content={"detail": "Remote API access requires SMART_RESEARCH_API_KEY to be configured."},
            )

    now = time.monotonic()
    with _RATE_LIMIT_LOCK:
        if client_host not in _RATE_LIMIT_BUCKETS and len(_RATE_LIMIT_BUCKETS) >= 4096:
            expired = [
                host
                for host, stamps in _RATE_LIMIT_BUCKETS.items()
                if not stamps or now - stamps[-1] >= _RATE_LIMIT_WINDOW_SECONDS
            ]
            for host in expired:
                _RATE_LIMIT_BUCKETS.pop(host, None)
            if len(_RATE_LIMIT_BUCKETS) >= 4096:
                return JSONResponse(status_code=429, content={"detail": "API rate limiter is at capacity; retry shortly."})

        bucket = _RATE_LIMIT_BUCKETS.setdefault(client_host, [])
        bucket[:] = [stamp for stamp in bucket if now - stamp < _RATE_LIMIT_WINDOW_SECONDS]
        if len(bucket) >= _RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "API request limit exceeded; retry shortly."})
        bucket.append(now)
    return await call_next(request)


def _run_pipeline_request(request: PipelineRequest) -> FullPipelineResponse:
    topic = sanitize_academic_topic(request.topic)
    domain = request.domain or infer_domain(topic, request.domain, request.objective)
    objective = request.objective or "Design a rigorous, reproducible, and evidence-driven study."
    if request.require_verified_data:
        validate_verified_request(topic, request.domain, request.objective)
    if not request.api_key:
        load_dotenv(_REPO_ROOT / ".env", override=True)

    api_key = request.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = request.model or os.getenv("GEMINI_MODEL") or None

    plan = maybe_live_plan(topic, domain, objective, request.constraints, api_key, model=model)
    mode_used = "live_gemini" if plan is not None else "dynamic_fallback"
    if plan is None:
        plan = build_dynamic_planner(topic, domain, objective)
    plan = ensure_system_intelligence(plan, topic, domain, objective)

    research = build_dynamic_research(plan)
    validate_research_contract(plan, research)

    live_paper = maybe_live_writer(plan, research, api_key, model=model)
    paper = live_paper or build_synthesis_writer(plan, research)
    paper = sanitize_writer_output(paper)
    paper = ensure_gap_and_proposal_sections(paper, plan)
    paper.abstract = academic_abstract(paper.abstract)
    paper.mathematical_formulas = paper.mathematical_formulas[:4]

    guardrail = GuardrailOutput.model_validate(
        audit_manuscript(paper.model_dump(), research.model_dump())
    )
    review = build_dynamic_reviewer(paper, research)
    return FullPipelineResponse(
        mode_used=mode_used,
        paper_mode="live_gemini" if live_paper else "dynamic_fallback",
        plan=plan,
        research=research,
        paper=paper,
        review=review,
        guardrail=guardrail,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smart-research-lab-api"}


@app.post("/api/planner/generate", response_model=PlannerOutput)
def generate_planner_plan(payload: TopicInput) -> PlannerOutput:
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    request = PipelineRequest(topic=payload.topic, domain=payload.domain, objective=payload.objective)
    return _run_pipeline_request(request).plan


@app.post("/api/researcher/generate", response_model=ResearcherOutput)
def generate_research_plan(payload: PlannerOutput) -> ResearcherOutput:
    return build_dynamic_research(payload)


@app.post("/api/writer/generate", response_model=WriterOutput)
def generate_writer_output(payload: dict[str, Any]) -> WriterOutput:
    plan = PlannerOutput.model_validate(payload.get("plan", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return build_synthesis_writer(plan, research)


@app.post("/api/reviewer/generate", response_model=ReviewerOutput)
def generate_reviewer_output(payload: dict[str, Any]) -> ReviewerOutput:
    paper = WriterOutput.model_validate(payload.get("paper", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return build_dynamic_reviewer(paper, research)


@app.post("/api/paper/ask")
def ask_paper_question(payload: dict[str, Any]) -> dict[str, str]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    paper = WriterOutput.model_validate(payload.get("paper", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return {"answer": _answer_paper_question(question, paper, research)}


@app.post("/api/pipeline/run", response_model=FullPipelineResponse)
def run_pipeline(payload: PipelineRequest) -> FullPipelineResponse:
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    return _run_pipeline_request(payload)


__all__ = [
    "PTIT_COURSE_METADATA",
    "_EXHAUSTED_MODELS",
    "_answer_paper_question",
    "_run_pipeline_request",
    "_safe_live_call",
    "app",
]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
