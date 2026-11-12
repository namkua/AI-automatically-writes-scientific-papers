from __future__ import annotations

from frontend.exporters import pdf_report, persist_artifacts, sanitize_paper_data
from frontend.ui import main
from frontend.views import render_benchmarks

__all__ = [
    "main",
    "pdf_report",
    "persist_artifacts",
    "render_benchmarks",
    "sanitize_paper_data",
]

if __name__ == "__main__":
    main()
