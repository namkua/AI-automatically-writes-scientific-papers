from __future__ import annotations

import base64
import gc
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from backend.sanitizer import strip_math_delimiters
from frontend.api_client import answer_paper_question
from frontend.exporters import sanitize_paper_data
from frontend.styles import display_frame, metric_explanation


def _render_pdf_open_button(pdf: bytes) -> None:
    encoded_pdf = base64.b64encode(pdf).decode("ascii")
    component_html = f"""
    <button id="open-pdf" type="button">🔍 Mở PDF ở tab mới</button>
    <script>
      const encodedPdf = "{encoded_pdf}";
      document.getElementById("open-pdf").addEventListener("click", () => {{
        const binary = atob(encodedPdf);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {{
          bytes[index] = binary.charCodeAt(index);
        }}
        const url = URL.createObjectURL(new Blob([bytes], {{ type: "application/pdf" }}));
        window.open(url, "_blank", "noopener,noreferrer");
        window.setTimeout(() => URL.revokeObjectURL(url), 60000);
      }});
    </script>
    <style>
      body {{ margin: 0; background: transparent; }}
      #open-pdf {{
        width: 100%; min-height: 3rem; border: 0; border-radius: .5rem;
        background: #d95757; color: white; cursor: pointer;
        font: 800 14px sans-serif;
      }}
      #open-pdf:hover {{ background: #ee6c6c; }}
    </style>
    """
    import streamlit.components.v1 as components

    components.html(component_html, height=60, scrolling=False)


def render_header(result: dict[str, Any]) -> None:
    plan, research, review = result["plan"], result["research"], result["review"]
    proposed = next(item for item in research["models_compared"] if item["role"] == "Proposed")
    proposed_row = next(item for item in research["benchmark_table"] if item["role"] == "Proposed")
    primary_metric = metric_explanation(research)
    metric_name = primary_metric.split(" — ", 1)[0]
    st.markdown(
        f"<section class='hero'><h1>{escape(plan['title'])}</h1>"
        "<p>Phòng thí nghiệm nghiên cứu AI · Hệ thống đa tác tử</p>"
        f"<span class='domain-badge'>{escape(plan['domain'])}</span></section>",
        unsafe_allow_html=True,
    )
    metric_rows = [st.columns(3), st.columns(3)]
    cards = metric_rows[0] + metric_rows[1]
    metric_cards = [
        (f"Primary Score ({metric_name})", "Điểm chất lượng chính", f"{proposed_row['primary_metric']:.2f}", "Higher is better · simulation"),
        ("Latency", "Thời gian phản hồi, ms", f"{proposed['latency_ms']:.1f} ms", f"{proposed['params_m']:.1f}M parameters"),
        ("Experiment Scope", "Quy mô thử nghiệm", f"{len(research['datasets_used'])} datasets", "Evaluated datasets"),
        (
            "Baseline Comparison" if research.get("baseline_comparison") else "Review Score",
            "Nhận xét PDF gốc" if research.get("baseline_comparison") else "Điểm phản biện",
            ("AI notes ready" if research.get("baseline_comparison", {}).get("review") else "Table ready")
            if research.get("baseline_comparison")
            else f"{review['overall_score']:.1f} / 10",
            "Direct comparison with uploaded PDF" if research.get("baseline_comparison") else f"Structural review: {review['verdict']}",
        ),
        ("Reasoning Consistency", "Độ nhất quán suy luận", f"{research.get('reasoning_consistency_score', 0):.2f}", "Routing logic audit"),
        ("Knowledge Precision", "Độ chính xác tri thức", f"{research.get('knowledge_retrieval_precision', 0):.2f}", "Knowledge catalog match"),
    ]
    for column, (title, title_vi, value, badge) in zip(cards, metric_cards):
        with column:
            st.markdown(
                f"<div class='metric-card'><div class='metric-card-title'>{escape(title)}<span>({escape(title_vi)})</span></div>"
                f"<div class='metric-card-value'>{escape(value)}</div>"
                f"<div class='metric-card-badge'>{escape(badge)}</div></div>",
                unsafe_allow_html=True,
            )


def render_plan(plan: dict[str, Any]) -> None:
    st.markdown(
        f"<div class='callout'><strong>Research Question</strong><br>{escape(plan['research_question'])}<br><br>"
        f"<strong>Core Hypothesis</strong><br>{escape(plan['hypothesis'])}</div>",
        unsafe_allow_html=True,
    )
    architecture = st.columns(3)
    architecture[0].markdown("**Knowledge Base (Kho tri thức)**\n\n" + "\n".join(f"- {item}" for item in plan.get("knowledge_base", {}).get("taxonomy", [])))
    architecture[1].markdown(
        "**Inference Engine (Bộ suy luận)**\n\n"
        + plan.get("inference_engine", {}).get("matching_strategy", "Not specified")
        + "\n\nDomain (Lĩnh vực): "
        + plan.get("inference_engine", {}).get("selected_domain", "Not specified")
    )
    architecture[2].markdown("**Multi-Agent Coordination (Phối hợp đa tác tử)**\n\n" + "\n".join(f"- {item}" for item in plan.get("agent_coordination", {}).get("agents", [])))

    st.subheader("📦 Datasets Catalog (Danh mục tập dữ liệu)")
    columns = st.columns(3)
    for index, dataset in enumerate(plan["datasets_catalog"]):
        with columns[index % 3]:
            st.markdown(
                f"<article class='glass-card'><h4>{escape(dataset['name'])}</h4>"
                f"<span class='badge'>{escape(dataset['modality'])}</span>"
                f"<p><b>Scale (Quy mô):</b> {escape(dataset['sample_size'])}<br>"
                f"<b>Input (Đầu vào):</b> {escape(dataset['input_spec'])}<br>"
                f"<b>Metric (Chỉ số):</b> {escape(dataset['primary_metric'])}</p>"
                f"<small>{escape(dataset['description'])}</small></article>",
                unsafe_allow_html=True,
            )

    st.subheader("🧠 Model & Architecture Catalog (Danh mục mô hình và kiến trúc)")
    models = pd.DataFrame([item for item in plan["models_catalog"] if item["role"] == "Proposed"])
    st.dataframe(display_frame(models, ["name", "backbone", "params_m", "flops_g", "latency_ms", "key_characteristic"]), width="stretch", hide_index=True)
    if HAS_PLOTLY:
        chart = models.melt(id_vars="name", value_vars=["params_m", "latency_ms"], var_name="Measure", value_name="Value")
        chart["Measure"] = chart["Measure"].map({
            "params_m": "Parameters (Tham số, M)",
            "latency_ms": "Latency (Độ trễ, ms)",
        }).fillna(chart["Measure"])
        st.plotly_chart(
            px.bar(
                chart,
                x="name",
                y="Value",
                color="Measure",
                barmode="group",
                template="plotly_dark",
                title="Parameters vs latency (Tham số và độ trễ)",
                labels={"name": "Model (Mô hình)", "Value": "Value (Giá trị)", "Measure": "Measure (Chỉ số)"},
            ),
            width="stretch",
        )


def render_benchmarks(research: dict[str, Any]) -> None:
    primary_metric = metric_explanation(research)
    curves = pd.DataFrame(research["training_curves"])
    left, right = st.columns(2)
    if HAS_PLOTLY and not curves.empty:
        color_col = "dataset_name" if "dataset_name" in curves.columns else None
        legend_layout = {"height": 360, "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}}
        loss_fig = px.line(
            curves,
            x="epoch",
            y="proposed_loss",
            color=color_col,
            markers=True,
            template="plotly_dark",
            title="Validation Loss (Loss xác thực)",
            labels={
                "epoch": "Epoch (Vòng huấn luyện)",
                "proposed_loss": "Validation Loss (Sai số dự đoán; thấp hơn tốt hơn)",
                "dataset_name": "Dataset (Tập dữ liệu)",
            },
            hover_data={"epoch": True, "proposed_loss": ":.4f"},
        )
        loss_fig.update_layout(**legend_layout)
        left.plotly_chart(loss_fig, width="stretch")

        score_fig = px.line(
            curves,
            x="epoch",
            y="proposed_score",
            color=color_col,
            markers=True,
            template="plotly_dark",
            title=f"Primary Metric (Chỉ số chính: {primary_metric})",
            labels={
                "epoch": "Epoch (Vòng huấn luyện)",
                "proposed_score": f"Primary Metric (Chỉ số chính: {primary_metric})",
                "dataset_name": "Dataset (Tập dữ liệu)",
            },
            hover_data={"epoch": True, "proposed_score": ":.4f"},
        )
        score_fig.update_layout(**legend_layout)
        right.plotly_chart(score_fig, width="stretch")

    st.subheader("So sánh các mô hình trong benchmark nội bộ")
    benchmark = pd.DataFrame(research["benchmark_table"])
    st.caption("Bảng và biểu đồ đối chiếu phương pháp đề xuất với các baseline do hệ thống tạo cho benchmark này. Đây không phải so sánh với PDF bài báo gốc đã upload; xem nhận xét bài mẫu ở tab Cấu hình chủ đề.")
    st.dataframe(display_frame(benchmark), width="stretch", hide_index=True)

    if HAS_PLOTLY:
        benchmark_chart = benchmark.copy()
        benchmark_chart["model_label"] = benchmark_chart["model_name"].map({
            "nanoGPT Baseline": "nanoGPT Base",
            "nanoGPT + QK-Norm": "+ QK-Norm",
            "nanoGPT + Layer-wise LR": "+ Layer LR",
        }).fillna(benchmark_chart["model_name"].str.slice(0, 24))
        benchmark_chart["role"] = benchmark_chart["role"].map({"Baseline": "Baseline", "Proposed": "Proposed"}).fillna(benchmark_chart["role"])

        metric_fig = px.bar(
            benchmark_chart,
            x="model_label",
            y="primary_metric",
            color="role",
            title="Primary metric",
            labels={"model_label": "Model", "primary_metric": primary_metric.split(" — ", 1)[0], "role": "Type"},
            hover_data={"model_name": True, "primary_metric": ":.4f"},
            template="plotly_dark",
        )
        latency_fig = px.bar(
            benchmark_chart,
            x="model_label",
            y="latency_ms",
            color="role",
            title="Latency (ms)",
            labels={"model_label": "Model", "latency_ms": "Latency (ms)", "role": "Type"},
            hover_data={"model_name": True, "latency_ms": ":.2f"},
            template="plotly_dark",
        )
        params_fig = px.bar(
            benchmark_chart,
            x="model_label",
            y="params_m",
            color="role",
            title="Parameters (millions)",
            labels={"model_label": "Model", "params_m": "Parameters (M)", "role": "Type"},
            hover_data={"model_name": True, "params_m": ":.2f"},
            template="plotly_dark",
        )
        for figure in (metric_fig, latency_fig, params_fig):
            figure.update_layout(
                height=360,
                margin={"l": 55, "r": 20, "t": 55, "b": 65},
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            )
            figure.update_xaxes(tickangle=-20, automargin=True)
            figure.update_yaxes(automargin=True)
        metric_column, latency_column = st.columns(2)
        metric_column.plotly_chart(metric_fig, width="stretch")
        latency_column.plotly_chart(latency_fig, width="stretch")
        st.plotly_chart(params_fig, width="stretch")

    if research.get("ablation_study"):
        st.subheader("Thử nghiệm loại bỏ từng thành phần")
        ablation = pd.DataFrame(research["ablation_study"])
        st.dataframe(display_frame(ablation), width="stretch", hide_index=True)
        if HAS_PLOTLY:
            ablation_chart = ablation.copy()
            ablation_chart["configuration"] = ablation_chart["configuration"].map({
                "Full proposed model": "Full proposed model (Mô hình đầy đủ)",
            }).fillna(ablation_chart["configuration"])
            ablation_chart["removed_component"] = ablation_chart["removed_component"].map({
                "None": "None (Không bỏ thành phần)",
                "Adaptive gating": "Adaptive Gating (Cổng thích ứng)",
                "Robustness regularization": "Robustness Regularization (Điều chuẩn độ bền)",
                "Uncertainty calibration": "Uncertainty Calibration (Hiệu chỉnh độ tin cậy)",
            }).fillna(ablation_chart["removed_component"])
            st.plotly_chart(
                px.bar(
                    ablation_chart,
                    x="configuration",
                    y="primary_metric",
                    color="removed_component",
                    title=f"Effect of removing each component (Tác động khi bỏ từng thành phần: {primary_metric})",
                    labels={"configuration": "Configuration (Cấu hình)", "primary_metric": f"Primary Metric ({primary_metric})", "removed_component": "Removed Component (Thành phần bỏ)"},
                    hover_data={"configuration": True, "removed_component": True, "primary_metric": ":.4f"},
                    template="plotly_dark",
                ),
                width="stretch",
            )

    st.subheader("Phát hiện chính")
    for finding in research["key_findings"]:
        st.markdown(f"- {finding}")


def render_verified_datasets() -> None:
    st.subheader("Verified Local Datasets")
    st.caption("Only files present on this machine are marked Ready. Catalog-only datasets are never treated as measured evidence.")
    data_root = Path(__file__).resolve().parent.parent / "data"
    entries = [
        ("AG News", data_root / "huggingface" / "ag_news" / "train.jsonl", "Text classification", "https://huggingface.co/datasets/fancyzhx/ag_news"),
        ("IMDb Large Movie Review", data_root / "huggingface" / "imdb" / "train.jsonl", "Sentiment classification", "https://huggingface.co/datasets/stanfordnlp/imdb"),
    ]
    rows = []
    for name, path, task, source in entries:
        ready = path.is_file() and path.stat().st_size > 0
        rows.append({"Dataset": name, "Task": task, "Status": "Ready" if ready else "Not downloaded", "Size (MB)": round(path.stat().st_size / 1_000_000, 2) if ready else 0.0, "Source": source})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    available = [row["Dataset"] for row in rows if row["Status"] == "Ready"]
    if available:
        selected = st.selectbox("Inspect a local dataset", available, key="verified_dataset_select")
        selected_path = next(path for name, path, _, _ in entries if name == selected)
        if selected_path.suffix == ".jsonl":
            sample = pd.read_json(selected_path, lines=True).head(20)
            st.dataframe(sample, width="stretch", hide_index=True)
    else:
        st.info("No verified local dataset is available yet. Run the preparation command shown below.")
    st.markdown(
        '<div class="dataset-command"><code>python '
        '<span class="path-command">data/prepare_hf_text.py</span> '
        '<span class="flag-command">--dataset</span> ag_news</code></div>',
        unsafe_allow_html=True,
    )


def render_method(plan: dict[str, Any], paper: dict[str, Any]) -> None:
    paper = sanitize_paper_data(paper)
    st.subheader("Luồng kiến trúc")
    flow_steps = [
        "Input data",
        "Domain encoder",
        "Task-aware signal selection",
        "Proposed model",
        "Prediction + uncertainty",
    ]
    flow_html = '<div class="architecture-flow">'
    for index, step in enumerate(flow_steps):
        if index:
            flow_html += '<span class="flow-arrow" aria-hidden="true">→</span>'
        flow_html += f'<span class="flow-step">{escape(step)}</span>'
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)
    st.write(paper["methodology"])
    st.subheader("Mathematical Formulation")
    for formula in paper["mathematical_formulas"]:
        cleaned_formula = strip_math_delimiters(formula)
        if cleaned_formula:
            st.latex(cleaned_formula)
    st.subheader(f"PyTorch Core: {plan['proposed_method']}")
    pytorch_code = """class TaskAwareBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.sigmoid(self.gate(self.norm(x)))
        return x + weights * self.projection(self.norm(x))
"""
    st.markdown(
        f'<div class="architecture-code"><pre>{escape(pytorch_code)}</pre></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _render_pdf_preview_pages(pdf_bytes: bytes, max_pages: int = 3) -> tuple[list[bytes], int]:
    """Convert PDF pages to PNG bytes and cache them to avoid re-render on every Streamlit rerun."""
    import pymupdf as fitz

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = document.page_count
    images: list[bytes] = []
    for page_number, page in enumerate(document, 1):
        if page_number > max_pages:
            break
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
        images.append(pixmap.tobytes("png"))
    document.close()
    gc.collect()
    return images, total_pages


def render_paper(paper: dict[str, Any], research: dict[str, Any]) -> None:
    paper = sanitize_paper_data(paper)
    pdf = st.session_state["pdf_bytes"]
    st.markdown(
        f"<section class='hero'><h1>{escape(paper['title'])}</h1>"
        "<p>Bản thảo PDF đã sẵn sàng · Bố cục học thuật</p>"
        f"<span class='domain-badge'>{escape(str(research.get('inference_engine', {}).get('selected_domain', 'AI Research')))}</span></section>",
        unsafe_allow_html=True,
    )
    action_columns = st.columns(3)
    with action_columns[0]:
        st.download_button(
            "📥 Tải bài báo PDF",
            pdf,
            "research_paper.pdf",
            "application/pdf",
            key="download_paper_pdf_top",
            width="stretch",
        )
    with action_columns[1]:
        _render_pdf_open_button(pdf)
    with action_columns[2]:
        st.download_button(
            "📦 Tải toàn bộ artifacts (.ZIP)",
            st.session_state["zip_bytes"],
            "research_artifacts.zip",
            "application/zip",
            key="download_research_zip_top",
            width="stretch",
        )

    st.markdown("#### Xem trước bản thảo")
    comparison = research.get("baseline_comparison", {})
    if comparison.get("rows"):
        st.subheader("Baseline Paper Comparison")
        st.warning(comparison.get("status", "Provisional comparison"))
        st.dataframe(pd.DataFrame(comparison["rows"]).fillna("").astype(str), width="stretch", hide_index=True)
    try:
        images, total_pages = _render_pdf_preview_pages(pdf, max_pages=3)
        st.session_state["pdf_pages_images"] = images
        for page_number, page_image in enumerate(images, 1):
            st.image(page_image, width="stretch")
            st.caption(f"Trang {page_number}")
        if total_pages > 3:
            st.caption(f"Xem trước 3/{total_pages} trang. Tải file PDF bên dưới để xem toàn bộ.")
    except (ImportError, RuntimeError, ValueError) as error:
        st.warning(f"Không thể dựng ảnh xem trước PDF: {error}. Hãy tải file PDF bên dưới.")

    st.divider()
    st.subheader("💬 Hỏi về bài báo")
    sample_prompts = [
        ("📌", "What is this paper about?"),
        ("💡", "What is the main idea?"),
        ("📊", "What are the main results?"),
        ("❓", "What is the problem solved?"),
    ]
    sample_cols = st.columns(4)
    for idx, (emoji, prompt_text) in enumerate(sample_prompts):
        with sample_cols[idx]:
            if st.button(f"{emoji} {prompt_text}", key=f"sample_q_{idx}", width="stretch"):
                st.session_state["pending_sample_question"] = prompt_text
    if "pending_sample_question" in st.session_state:
        st.session_state["paper_question"] = st.session_state.pop("pending_sample_question")
    question = st.text_input("Đặt câu hỏi về bài báo", placeholder="Ví dụ: Bài báo so sánh với baseline nào?", key="paper_question")
    if question.strip():
        history = st.session_state.setdefault("paper_chat_history", [])
        if st.session_state.get("paper_question_answered") != question or not history:
            st.session_state["paper_question_answered"] = question
            with st.spinner("Đang phân tích bản thảo..."):
                answer = answer_paper_question(question, paper, research)
            if not history or history[-1]["question"] != question:
                history.append({"question": question, "answer": answer})
    for turn in reversed(st.session_state.get("paper_chat_history", [])):
        st.markdown(
            f"<div class='chat-response'><strong>Q: {escape(turn['question'])}</strong><br><br>"
            f"<strong>Research Assistant</strong><br>{escape(turn['answer'])}</div>",
            unsafe_allow_html=True,
        )


def render_artifacts() -> None:
    artifact_dir = Path(st.session_state["artifact_dir"])
    st.subheader("📦 Kho tệp kết quả")
    st.caption("Các tệp được giữ trong phiên làm việc; chuyển tab hoặc tải xuống không làm mất kết quả.")
    files = [
        ("📄 PDF Manuscript", "pdf_bytes", "application/pdf", "research_paper.pdf"),
        ("🏛️ IEEE 2-Column HTML", "html_bytes", "text/html", "paper_ieee.html"),
        ("🧾 LaTeX Source", "tex_bytes", "text/plain", "paper.tex"),
        ("📚 BibTeX References", "bib_bytes", "text/plain", "references.bib"),
        ("📝 Markdown Source", "md_bytes", "text/markdown", "paper.md"),
        ("📊 Benchmark JSON", "benchmark_bytes", "application/json", "benchmark_matrix.json"),
        ("🛡️ Guardrail Report", "guardrail_bytes", "application/json", "guardrail_report.json"),
        ("📦 Full Bundle (.ZIP)", "zip_bytes", "application/zip", "research_artifacts.zip"),
    ]
    cards = st.columns(4)
    for index, (label, state_key, mime, filename) in enumerate(files):
        with cards[index % len(cards)]:
            st.markdown(f"<article class='artifact-card'><strong>{label}</strong><br><small>{filename}</small></article>", unsafe_allow_html=True)
            st.download_button(f"Tải {filename}", st.session_state.get(state_key, b""), filename, mime, key=f"download_{state_key}", width="stretch")
    st.divider()
    st.caption(f"Thư mục kết quả: {artifact_dir.name}")


def render_review(review: dict[str, Any]) -> None:
    metrics = st.columns(4)
    for column, label, key in zip(
        metrics,
        ("Novelty (Tính mới)", "Rigor (Độ chặt chẽ)", "Clarity (Độ rõ ràng)", "Overall Score (Điểm tổng)"),
        ("novelty_score", "rigor_score", "clarity_score", "overall_score"),
    ):
        column.metric(label, f"{review[key]:.1f} / 10")
    st.markdown(f"<div class='callout'><strong>Verdict (Đánh giá): {escape(str(review['verdict']))}</strong></div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for column, title, key in zip(
        columns,
        ("✅ Strengths (Điểm mạnh)", "⚠️ Weaknesses (Điểm yếu)", "🛠️ Suggested Revisions (Đề xuất sửa)"),
        ("strengths", "weaknesses", "revision_suggestions"),
    ):
        with column:
            st.subheader(title)
            for item in review[key]:
                st.markdown(f"- {item}")


def render_direct_baseline_comparison(research: dict[str, Any]) -> None:
    comparison = research.get("baseline_comparison")
    if not comparison:
        return
    st.subheader("So sánh trực tiếp với PDF bài gốc")
    st.dataframe(pd.DataFrame(comparison.get("rows", [])).fillna("").astype(str), width="stretch", hide_index=True)
    notes = comparison.get("review", {})
    if notes:
        st.markdown("### Nhận xét ngắn")
        cards = st.columns(3)
        for column, key, heading in zip(
            cards,
            ("strengths", "limitations", "suggestions"),
            ("🎯 Điểm mạnh cốt lõi", "⚠️ Hạn chế chính", "🛠️ Đề xuất cải tiến"),
        ):
            item = notes.get(key, {})
            with column:
                st.markdown(f"**{heading}**")
                st.write(item.get("text", "Chưa đủ bằng chứng để nhận xét."))
        with st.expander("Xem câu dẫn dùng để kiểm chứng nhận xét"):
            for key, heading in (("strengths", "Điểm mạnh"), ("limitations", "Hạn chế"), ("suggestions", "Đề xuất")):
                item = notes.get(key, {})
                st.markdown(f"**{heading} · PDF gốc:** {item.get('original_quote', '')}")
                st.markdown(f"**{heading} · Bài mới:** {item.get('proposed_quote', '')}")
    else:
        st.warning("Chưa tạo được nhận xét AI có đủ câu dẫn kiểm chứng. Bảng trên chỉ hiển thị dữ liệu đã trích xuất.")
    st.caption(comparison.get("status", ""))

