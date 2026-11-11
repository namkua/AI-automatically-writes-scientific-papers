from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def generate_research_charts(
    research: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Create publication-ready benchmark and ablation charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    gc.collect()
    
    rows = pd.DataFrame(research.get("benchmark_table", []))
    created: list[Path] = []

    if not rows.empty:
        try:
            for column in ("latency_ms", "vram_mb"):
                if column not in rows:
                    rows[column] = 0.0
                rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0)
            labels = rows["model_name"].astype(str).tolist()
            short_labels = [label[:24] + ("…" if len(label) > 24 else "") for label in labels]
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=False)
            axes[0].bar(short_labels, rows["latency_ms"], color="#2563eb")
            axes[0].set_title("Latency (ms) - lower is better")
            axes[0].set_ylabel("Milliseconds")
            axes[1].bar(short_labels, rows["vram_mb"], color="#14b8a6")
            axes[1].set_title("Peak VRAM (MB) - lower is better")
            axes[1].set_ylabel("MB")
            for axis in axes:
                axis.tick_params(axis="x", labelrotation=28, labelsize=7)
                axis.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = destination / "benchmark_efficiency.png"
            figure.savefig(path, dpi=150, bbox_inches=None)
            plt.close(figure)
            created.append(path)
        except Exception:
            plt.close("all")
            gc.collect()

    ablations = pd.DataFrame(research.get("ablation_study", []))
    if not ablations.empty:
        try:
            ablations = ablations.copy()
            ablations["configuration"] = ablations["configuration"].astype(str).str.slice(0, 24)
            ablations["removed_component"] = ablations["removed_component"].astype(str).str.slice(0, 24)
            figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=False)
            axis.plot(
                ablations["configuration"],
                ablations["primary_metric"],
                marker="o",
                linewidth=2,
                label="Primary metric",
            )
            axis.set_title("Ablation trade-off")
            axis.set_ylabel("Primary metric")
            axis.tick_params(axis="x", labelrotation=28, labelsize=7)
            axis.grid(alpha=0.25)
            axis.legend()
            plt.tight_layout()
            path = destination / "ablation_tradeoff.png"
            figure.savefig(path, dpi=150, bbox_inches=None)
            plt.close(figure)
            created.append(path)
        except Exception:
            plt.close("all")
            gc.collect()

    gc.collect()
    plt.close("all")
    return created
