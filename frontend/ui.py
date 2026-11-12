from __future__ import annotations

import gc
import hashlib
import json
import os
from html import escape
from pathlib import Path

import streamlit as st

from frontend.api_client import run_pipeline
from frontend.baseline_comparison import (
    build_grounded_baseline_comparison,
    generate_baseline_review_notes,
    render_baseline_controls,
)
from frontend.exporters import persist_artifacts
from frontend.styles import CUSTOM, CUSTOM_CSS, PRESETS, apply_preset
from frontend.views import (
    render_artifacts,
    render_benchmarks,
    render_direct_baseline_comparison,
    render_header,
    render_method,
    render_paper,
    render_plan,
    render_review,
    render_verified_datasets,
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except Exception:
        pass


def main() -> None:
    """Entry point for the Streamlit Smart Research Lab web interface."""
    st.set_page_config(
        page_title="Phòng thí nghiệm nghiên cứu AI",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _load_env()

    first = next(iter(PRESETS))
    for key, value in {"topic": first, **PRESETS[first]}.items():
        st.session_state.setdefault(key, value)
    for key, value in {
        "full_pipeline_result": None,
        "full_paper_text": "",
        "pdf_file_path": "",
        "benchmark_data": [],
        "pdf_bytes": b"",
        "html_bytes": b"",
        "tex_bytes": b"",
        "bib_bytes": b"",
        "md_bytes": b"",
        "benchmark_bytes": b"",
        "guardrail_bytes": b"",
        "zip_bytes": b"",
        "artifact_dir": "",
        "baseline_metadata": None,
        "baseline_pdf_text": "",
        "baseline_pdf_signature": None,
        "baseline_comparison_token": None,
        "gemini_api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        "gemini_model": os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
        "paper_chat_history": [],
    }.items():
        st.session_state.setdefault(key, value)

    for key, default in (
        ("preset", CUSTOM),
        ("topic", ""),
        ("domain", ""),
        ("objective", ""),
        ("constraints", ""),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("## Bảng điều khiển nghiên cứu")
        _load_env()
        env_gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        entered_key = st.text_input(
            "🔑 Gemini API Key (tùy chọn)",
            value="",
            placeholder="Đang dùng key từ file .env" if env_gemini_key else "Dán API Key vào đây (hoặc đặt trong .env)",
            type="password",
            key="gemini_api_key_input",
            help="Dán Google Gemini API Key để kích hoạt Gemini Live Mode. Nếu để trống, hệ thống tự động đọc key từ file .env (nếu có) hoặc chạy Dynamic Fallback.",
        )
        effective_key = entered_key.strip() or env_gemini_key
        st.session_state["gemini_api_key"] = effective_key
        if effective_key:
            if os.environ.get("GEMINI_API_KEY") != effective_key:
                os.environ["GEMINI_API_KEY"] = effective_key
                try:
                    from backend.main import _EXHAUSTED_MODELS

                    _EXHAUSTED_MODELS.clear()
                except Exception:
                    pass
            model_choices = [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            selected_model = st.selectbox(
                "🤖 Chọn Gemini Model (ưu tiên trước)",
                model_choices,
                index=0,
                key="gemini_model_selector",
                help="Chọn mô hình Gemini ưu tiên; hệ thống tự động chuyển sang model dự phòng nếu model chính quá tải hoặc hết quota.",
            )
            st.session_state["gemini_model"] = selected_model
            os.environ["GEMINI_MODEL"] = selected_model
            source_note = "từ ô nhập" if entered_key.strip() else "từ file .env"
            st.caption(f"🟢 **Live Gemini ({selected_model})** — Key {source_note}")
        else:
            st.caption("⚪ **Dynamic Fallback** — Chưa có API Key")

        st.selectbox("⚡ Chủ đề mẫu", [*PRESETS, CUSTOM], key="preset", on_change=apply_preset)
        st.text_area("Chủ đề nghiên cứu", key="topic", height=90)
        st.text_input("Lĩnh vực", key="domain")
        st.text_input("Mục tiêu chính", key="objective")
        st.text_area("Ràng buộc", key="constraints", height=75)
        compare_clicked = render_baseline_controls()
        run_clicked = st.button("🚀 Bắt đầu sinh bài báo", type="primary", width="stretch")

    if run_clicked:
        st.cache_data.clear()
        st.cache_resource.clear()
        for heavy_key in (
            "full_pipeline_result",
            "full_paper_text",
            "pdf_bytes",
            "tex_bytes",
            "benchmark_bytes",
            "guardrail_bytes",
            "zip_bytes",
            "artifact_dir",
            "pdf_file_path",
            "benchmark_data",
            "baseline_metadata",
            "baseline_pdf_text",
            "baseline_pdf_signature",
            "baseline_comparison_token",
            "pdf_pages_images",
        ):
            st.session_state.pop(heavy_key, None)
        gc.collect()

        with st.status("🤖 Đang điều phối hệ thống nghiên cứu AI...", expanded=True) as status:
            status.write("⏳ Đang khởi tạo Planner Agent và phân loại chủ đề...")
            result = run_pipeline(
                st.session_state.topic,
                st.session_state.domain,
                st.session_state.objective,
                st.session_state.constraints,
            )
            gc.collect()
            if not result:
                status.update(label="Pipeline unavailable", state="error")
                st.stop()
            if "error" in result:
                status.update(label="Request rejected", state="error", expanded=True)
                st.error(result["error"])
                st.stop()
            status.write("✍️ Writer Agent đã hoàn tất bản thảo học thuật...")
            status.write("🛡️ Guardrail đang kiểm tra công thức, benchmark và evidence labels...")
            status.write("📄 Đang biên dịch PDF và đóng gói artifacts...")
            export_research = result["research"]
            paper = result["paper"]
            zip_bytes, artifact_dir = persist_artifacts(paper, export_research)
            pdf = (artifact_dir / "paper.pdf").read_bytes()
            st.session_state.full_pipeline_result = result
            st.session_state.full_paper_text = (artifact_dir / "paper.md").read_text(encoding="utf-8")
            st.session_state.pdf_bytes = pdf
            st.session_state.html_bytes = (artifact_dir / "paper_ieee.html").read_bytes()
            st.session_state.tex_bytes = (artifact_dir / "paper.tex").read_bytes()
            st.session_state.bib_bytes = (artifact_dir / "references.bib").read_bytes()
            st.session_state.md_bytes = (artifact_dir / "paper.md").read_bytes()
            st.session_state.zip_bytes = zip_bytes
            st.session_state.artifact_dir = str(artifact_dir)
            st.session_state.pdf_file_path = str(artifact_dir / "paper.pdf")
            st.session_state.benchmark_data = export_research.get("benchmark_table", [])
            st.session_state.benchmark_bytes = (artifact_dir / "benchmark_matrix.json").read_bytes()
            st.session_state.guardrail_bytes = (artifact_dir / "guardrail_report.json").read_bytes()
            st.session_state.paper_chat_history = []
            st.session_state.paper_question_answered = ""
            gc.collect()
            status.update(label="✅ Pipeline completed · manuscript ready", state="complete", expanded=False)

    result = st.session_state.get("full_pipeline_result")
    if not result:
        active_model = st.session_state.get("gemini_model") or "Dynamic Fallback"
        mode_badge = f"Live Gemini · {active_model}" if st.session_state.get("gemini_api_key") else "Dynamic Fallback"
        st.markdown(
            "<section class='hero'>"
            "<h1>Smart Research Lab · Hệ thống viết bài báo khoa học tự động</h1>"
            "<p>Sẵn sàng điều phối Planner, Researcher, Writer và Reviewer theo chủ đề bạn chọn.</p>"
            f"<span class='domain-badge'>{escape(mode_badge)}</span>"
            "</section>",
            unsafe_allow_html=True,
        )
        if compare_clicked:
            st.info("Vui lòng bấm **🚀 Bắt đầu sinh bài báo** trước để tạo bài mới rồi mới đối chiếu với PDF gốc.")
        st.info(
            "👈 Kiểm tra API Key, model Gemini và nhập **Chủ đề nghiên cứu** ở thanh bên trái, sau đó bấm **🚀 Bắt đầu sinh bài báo** để bắt đầu."
        )
        return

    baseline_metadata = st.session_state.get("baseline_metadata")
    baseline_text = st.session_state.get("baseline_pdf_text", "")
    needs_initial_pipeline = bool(result and baseline_metadata and not st.session_state.get("baseline_comparison_token"))
    should_compare = bool(compare_clicked or run_clicked or needs_initial_pipeline)
    if baseline_metadata and result and should_compare:
        comparison_payload = json.dumps(
            [baseline_metadata, baseline_text, result.get("paper", {}).get("title", "")],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        comparison_token = hashlib.sha256(comparison_payload.encode("utf-8")).hexdigest()
        if compare_clicked:
            st.session_state["baseline_comparison_token"] = None
        if st.session_state.get("baseline_comparison_token") != comparison_token:
            with st.spinner("Đang đối chiếu PDF gốc với bản thảo mới..."):
                comparison = build_grounded_baseline_comparison(baseline_metadata, result.get("research", {}))
                if baseline_text:
                    comparison["review"] = generate_baseline_review_notes(
                        baseline_text,
                        result.get("paper", {}),
                        result.get("research", {}),
                        baseline_metadata,
                    )
                result["research"]["baseline_comparison"] = comparison
                st.session_state["baseline_comparison_token"] = comparison_token

    if result.get("paper_mode") == "dynamic_fallback" and result.get("mode_used") == "live_gemini":
        st.warning("Gemini produced a live plan, but the manuscript uses the fallback writer because the live response was invalid. Benchmarks remain Controlled Planning Simulation Estimates.")
    elif result.get("paper_mode") == "dynamic_fallback":
        st.warning("Simulation mode: benchmark, latency, and training curves are planning estimates, not measurements from a real dataset. Set GEMINI_API_KEY for live planning; validate all numbers with instrumented experiments before submission.")
    else:
        st.success("Live planner and writer are enabled with Gemini. Benchmark values still require validation with instrumented experiments before publication.")

    render_header(result)
    plan, research, paper, review = result["plan"], result["research"], result["paper"], result["review"]
    tabs = [
        "⚙️ Cấu hình chủ đề",
        "📄 Xem trước bản thảo",
        "📊 So sánh & biểu đồ",
        "📦 Tệp kết quả",
    ]
    selected_tab = st.radio(
        "Chọn tab",
        tabs,
        key="selected_tab",
        horizontal=True,
        label_visibility="collapsed",
    )
    tab_index = tabs.index(selected_tab)
    if tab_index == 0:
        render_plan(plan)
        render_method(plan, paper)
        render_verified_datasets()
        if research.get("baseline_comparison"):
            render_direct_baseline_comparison(research)
        else:
            render_review(review)
    elif tab_index == 1:
        render_paper(paper, research)
    elif tab_index == 2:
        render_benchmarks(research)
    else:
        render_artifacts()

