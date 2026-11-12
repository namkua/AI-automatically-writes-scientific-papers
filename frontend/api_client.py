from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from backend.sanitizer import sanitize_academic_topic

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_active_gemini_key() -> str | None:
    try:
        typed = str(st.session_state.get("gemini_api_key_input", "") or "").strip()
        if typed:
            return typed
        stored = str(st.session_state.get("gemini_api_key", "") or "").strip()
        if stored:
            return stored
    except Exception:
        pass
    load_dotenv(_REPO_ROOT / ".env", override=True)
    candidate = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return candidate or None


def run_pipeline(topic: str, domain: str, objective: str, constraints: str) -> dict[str, Any] | None:
    topic = sanitize_academic_topic(topic)
    effective_constraints = constraints or ""
    baseline = st.session_state.get("baseline_metadata")
    if isinstance(baseline, dict):
        b_datasets = ", ".join(baseline.get("datasets_used") or [])
        b_metrics = ", ".join(list((baseline.get("reported_metrics") or {}).keys())[:8])
        b_model = baseline.get("model_name") or ""
        extra_parts = []
        if b_model and b_model != "Not identified":
            extra_parts.append(f"Compare against baseline model: {b_model}")
        if b_datasets:
            extra_parts.append(f"Include baseline datasets: {b_datasets}")
        if b_metrics:
            extra_parts.append(f"Align primary evaluation metrics with baseline metrics: {b_metrics}")
        if extra_parts:
            effective_constraints = (
                (effective_constraints + "; " if effective_constraints.strip() else "")
                + "; ".join(extra_parts)
            )

    payload = {
        "topic": topic,
        "domain": domain,
        "objective": objective,
        "constraints": effective_constraints,
        "api_key": get_active_gemini_key(),
        "model": st.session_state.get("gemini_model") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
        "require_verified_data": False,
    }
    try:
        api_key = os.getenv("SMART_RESEARCH_API_KEY", "").strip()
        headers = {"X-API-Key": api_key} if api_key else {}
        response = requests.post(
            "http://127.0.0.1:8000/api/pipeline/run",
            json=payload,
            headers=headers,
            timeout=300,
        )
        if response.ok:
            return response.json()
        if response.status_code in (404, 502, 503):
            raise requests.RequestException(f"Foreign service on port 8000 returned HTTP {response.status_code}")
        try:
            return {"error": response.json().get("detail", "The request was rejected by the research validator.")}
        except ValueError:
            return {"error": f"The backend rejected the request with HTTP {response.status_code}."}
    except requests.RequestException:
        pass
    try:
        from backend.main import _run_pipeline_request
        from backend.schemas import PipelineRequest

        return _run_pipeline_request(PipelineRequest(**payload)).model_dump(mode="json")
    except Exception as error:
        detail = getattr(error, "detail", str(error))
        return {"error": str(detail)}


def answer_paper_question(question: str, paper: dict[str, Any], research: dict[str, Any]) -> str:
    payload = {"question": question, "paper": paper, "research": research}
    try:
        api_key = os.getenv("SMART_RESEARCH_API_KEY", "").strip()
        headers = {"X-API-Key": api_key} if api_key else {}
        response = requests.post(
            "http://127.0.0.1:8000/api/paper/ask",
            json=payload,
            headers=headers,
            timeout=15,
        )
        if response.ok:
            return response.json().get("answer", "No answer available.")
    except requests.RequestException:
        pass
    try:
        from backend.gemini_client import answer_paper_question as local_answer
        from backend.schemas import ResearcherOutput, WriterOutput

        return local_answer(
            question,
            WriterOutput.model_validate(paper),
            ResearcherOutput.model_validate(research),
        )
    except Exception:
        return "I could not analyze that question from the generated manuscript. Try asking about the method, baselines, datasets, results, or limitations."

