"""Log benchmark results to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import wandb

_ORT_OP_COLORS = {
    "Add": "#2ca02c",
    "Concat": "#bcbd22",
    "Conv": "#1f77b4",
    "MaxPool": "#17becf",
    "Mul": "#9467bd",
    "PRelu": "#ff7f0e",
    "Resize": "#e377c2",
    "Slice": "#d62728",
    "Clip": "#8c564b",
    "Split": "#7f7f7f",
}


# Keep in sync with geyikbench.benchmark._METRIC_SKIP_S / _steady_start_index.
_METRIC_SKIP_S = 1.0

_T_CRIT_95 = {
    2: 12.7062047364,
    3: 4.30265272991,
    4: 3.18244630528,
    5: 2.7764451052,
    6: 2.57058183564,
    7: 2.44691185114,
    8: 2.36462425159,
    9: 2.30600405321,
    10: 2.26215716274,
    15: 2.14478668792,
    20: 2.09302405441,
    30: 2.04522964213,
}


def _t_crit_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for df = n - 1."""
    if n < 2:
        return float("nan")
    return _T_CRIT_95.get(n, 1.96)


def _steady_start_index(latencies_ms: list[float], skip_s: float = _METRIC_SKIP_S) -> int:
    """Index of the first sample after ``skip_s`` seconds of measured latency."""
    if skip_s <= 0.0 or not latencies_ms:
        return 0
    budget_ms = skip_s * 1000.0
    cumulative_ms = 0.0
    for i, latency_ms in enumerate(latencies_ms):
        cumulative_ms += latency_ms
        if cumulative_ms >= budget_ms:
            start = i + 1
            return start if start < len(latencies_ms) else 0
    return 0


def _histogram_figure(
    values: list[float],
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> Any | None:
    """Build a matplotlib histogram figure, or None if *values* is empty."""
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=min(bins, max(1, len(values))), color="#2a6f97", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _ort_cumulative_figure(
    nodes: list[dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    ci_lo_key: str | None = None,
    ci_hi_key: str | None = None,
    trial_totals: list[float] | None = None,
) -> Any | None:
    """Op-colored cumulative mean vs layer index (optional 95% CI band)."""
    if not nodes:
        return None
    ys = [float(n.get(y_key) or 0.0) for n in nodes]
    ops = [str(n.get("op_name") or "") for n in nodes]
    xs = list(range(len(nodes)))
    colors = [_ORT_OP_COLORS.get(op, "#333333") for op in ops]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    i = 0
    while i < len(nodes):
        j = i + 1
        while j < len(nodes) and ops[j] == ops[i]:
            j += 1
        ax.axvspan(i - 0.5, j - 0.5, color=colors[i], alpha=0.08, linewidth=0, zorder=0)
        i = j

    stored_lo = [n.get(ci_lo_key) for n in nodes] if ci_lo_key else []
    stored_hi = [n.get(ci_hi_key) for n in nodes] if ci_hi_key else []

    def _finite_ci(v: Any) -> bool:
        return isinstance(v, (int, float)) and v == v

    has_stored_ci = bool(
        stored_lo
        and stored_hi
        and _finite_ci(stored_lo[-1])
        and _finite_ci(stored_hi[-1])
    )
    drew_ci = False
    lo: list[float] = []
    hi: list[float] = []
    if has_stored_ci:
        lo = [float(v) for v in stored_lo]
        hi = [float(v) for v in stored_hi]
    elif trial_totals and len(trial_totals) >= 2 and ys and ys[-1] > 0:
        n_tot = len(trial_totals)
        t_crit = _t_crit_95(n_tot)
        sem = statistics.stdev(trial_totals) / math.sqrt(n_tot)
        half_end = t_crit * sem
        lo = [y - half_end * (y / ys[-1]) for y in ys]
        hi = [y + half_end * (y / ys[-1]) for y in ys]
    if lo and hi:
        ax.fill_between(
            xs, lo, hi, facecolor="#4c4c4c", edgecolor="#4c4c4c",
            alpha=0.28, linewidth=0.8, zorder=1, label="95% CI",
        )
        ax.plot(xs, lo, color="#4c4c4c", linewidth=0.8, alpha=0.85, zorder=2)
        ax.plot(xs, hi, color="#4c4c4c", linewidth=0.8, alpha=0.85, zorder=2)
        drew_ci = True

    ax.plot(xs, ys, color="#222222", linewidth=1.0, alpha=0.85, zorder=2)
    ax.scatter(xs, ys, c=colors, s=14, zorder=3, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_xlim(-0.5, len(nodes) - 0.5)
    ax.set_ylim(bottom=0)

    present = [op for op in _ORT_OP_COLORS if op in set(ops)]
    handles: list[Any] = []
    if drew_ci:
        handles.append(Patch(facecolor="#4c4c4c", alpha=0.25, label="95% CI"))
    handles.extend(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ORT_OP_COLORS[op], markersize=8, label=op)
        for op in present
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(6, max(1, len(handles))),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


def _sample_vs_metric_figure(
    values: list[float],
    *,
    title: str,
    ylabel: str,
) -> Any | None:
    """Build a sample-index vs metric line/scatter figure, or None if empty."""
    if not values:
        return None
    steps = list(range(len(values)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, values, color="#2a6f97", linewidth=1.0, alpha=0.85)
    ax.scatter(steps, values, color="#01497c", s=12, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def summary_metrics_from_result(data: dict[str, Any]) -> dict[str, Any]:
    """Build the scalar summary dict used by individual benchmark W&B runs."""
    latencies = data.get("latency_ms", [])
    powers = data.get("power_w", [])
    temps = data.get("temperature_c", [])
    freqs_sm = data.get("frequency_mhz", [])
    freqs_mem = data.get("frequency_mem_mhz", [])
    freqs_graphics = data.get("frequency_graphics_mhz", [])
    freqs_video = data.get("frequency_video_mhz", [])
    start = _steady_start_index(latencies)
    steady_powers = powers[start:]
    steady_temps = temps[start:]
    steady_freqs_sm = freqs_sm[start:]
    steady_freqs_mem = freqs_mem[start:]
    steady_freqs_graphics = freqs_graphics[start:]
    steady_freqs_video = freqs_video[start:]

    return {
        "latency_ms/mean": data["latency_ms_mean"],
        "latency_ms/std": data["latency_ms_std"],
        "latency_ms/p50": data["latency_ms_p50"],
        "latency_ms/p95": data["latency_ms_p95"],
        "energy_mj/mean": data["energy_mj_mean"],
        "energy_mj/std": data["energy_mj_std"],
        "energy_mj/total": data["energy_mj_total"],
        "power_w/mean": data["power_w_mean"],
        "power_w/std": statistics.pstdev(steady_powers) if len(steady_powers) > 1 else 0.0,
        "temperature_c/mean": data["temperature_c_mean"],
        "temperature_c/std": statistics.pstdev(steady_temps) if len(steady_temps) > 1 else 0.0,
        "frequency_mhz/mean": data.get("frequency_mhz_mean", float("nan")),
        "frequency_mhz/std": statistics.pstdev(steady_freqs_sm) if len(steady_freqs_sm) > 1 else 0.0,
        "frequency_mem_mhz/mean": data.get("frequency_mem_mhz_mean", float("nan")),
        "frequency_mem_mhz/std": statistics.pstdev(steady_freqs_mem) if len(steady_freqs_mem) > 1 else 0.0,
        "frequency_graphics_mhz/mean": data.get(
            "frequency_graphics_mhz_mean", float("nan")
        ),
        "frequency_graphics_mhz/std": (
            statistics.pstdev(steady_freqs_graphics) if len(steady_freqs_graphics) > 1 else 0.0
        ),
        "frequency_video_mhz/mean": data.get("frequency_video_mhz_mean", float("nan")),
        "frequency_video_mhz/std": (
            statistics.pstdev(steady_freqs_video) if len(steady_freqs_video) > 1 else 0.0
        ),
        "throughput_inf_per_s": data["throughput_inf_per_s"],
        "steady_skip_iters": data.get("steady_skip_iters", start),
        "steady_iters": data.get("steady_iters", max(0, len(latencies) - start)),
    }


def populate_wandb_run(
    run: Any,
    data: dict[str, Any],
    *,
    output_dir: str | Path,
    run_name: str,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    """Attach benchmark summary metrics, sample table, and plot images to an open run.

    Uses the same ``latency_ms/mean``-style keys and histogram/series media as
    individual ``benchmark.py`` runs, so project/sweep tables stay comparable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summary_metrics_from_result(data)
    if extra_summary:
        summary.update(extra_summary)
    # Summary-only scalars: avoid wandb.log() of scalars (empty chart panels).
    run.summary.update(summary)

    latencies = data.get("latency_ms", [])
    powers = data.get("power_w", [])
    temps = data.get("temperature_c", [])
    freqs_sm = data.get("frequency_mhz", [])
    freqs_mem = data.get("frequency_mem_mhz", [])
    freqs_graphics = data.get("frequency_graphics_mhz", [])
    freqs_video = data.get("frequency_video_mhz", [])
    energies = data.get("energy_mj", [])
    n = len(latencies)

    table = wandb.Table(
        columns=[
            "step",
            "latency_ms",
            "energy_mj",
            "power_w",
            "temperature_c",
            "frequency_mhz",
            "frequency_mem_mhz",
            "frequency_graphics_mhz",
            "frequency_video_mhz",
        ],
    )
    for i in range(n):
        table.add_data(
            i,
            latencies[i],
            energies[i] if i < len(energies) else float("nan"),
            powers[i] if i < len(powers) else float("nan"),
            temps[i] if i < len(temps) else float("nan"),
            freqs_sm[i] if i < len(freqs_sm) else float("nan"),
            freqs_mem[i] if i < len(freqs_mem) else float("nan"),
            freqs_graphics[i] if i < len(freqs_graphics) else float("nan"),
            freqs_video[i] if i < len(freqs_video) else float("nan"),
        )
    run.summary["samples"] = table

    plot_specs = [
        ("latency_ms", latencies, latencies, "Latency (ms)"),
        ("energy_mj", energies, energies, "Energy (mJ)"),
        ("power_w", powers, powers, "Power (W)"),
        ("temperature_c", temps, temps, "Temperature (°C)"),
        ("frequency_mhz", freqs_sm, freqs_sm, "SM frequency (MHz)"),
    ]
    hist_dir = output_dir / "histograms"
    series_dir = output_dir / "series"
    hist_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    media_images: dict[str, Any] = {}
    for key, series_values, hist_values, axis_label in plot_specs:
        hist_fig = _histogram_figure(
            hist_values,
            title=f"{run_name} — {key}",
            xlabel=axis_label,
        )
        if hist_fig is not None:
            png_path = hist_dir / f"{key}.png"
            hist_fig.savefig(png_path, dpi=120)
            media_images[f"histograms/{key}"] = wandb.Image(
                str(png_path), caption=f"{key} histogram"
            )
            plt.close(hist_fig)

        series_fig = _sample_vs_metric_figure(
            series_values,
            title=f"{run_name} — {key} over samples",
            ylabel=axis_label,
        )
        if series_fig is not None:
            png_path = series_dir / f"{key}.png"
            series_fig.savefig(png_path, dpi=120)
            media_images[f"series/{key}"] = wandb.Image(
                str(png_path), caption=f"{key} vs sample"
            )
            plt.close(series_fig)

    ort_nodes = (data.get("ort_profiler") or {}).get("nodes") or []
    if ort_nodes:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        runtime_fig = _ort_cumulative_figure(
            ort_nodes,
            y_key="cum_ms_mean",
            ylabel="Cumulative runtime (ms)",
            title=f"{run_name} — ORT cumulative runtime",
            ci_lo_key="cum_ms_ci95_lo",
            ci_hi_key="cum_ms_ci95_hi",
            trial_totals=list(data.get("latency_ms") or []),
        )
        if runtime_fig is not None:
            png_path = plots_dir / "lop7_ort_profiler_index_vs_runtime.png"
            runtime_fig.savefig(png_path, dpi=150)
            media_images["plots/ort_cumulative_runtime"] = wandb.Image(
                str(png_path), caption="ORT cumulative runtime vs layer index"
            )
            plt.close(runtime_fig)
        energy_fig = _ort_cumulative_figure(
            ort_nodes,
            y_key="cum_mj_mean",
            ylabel="Cumulative energy (mJ)",
            title=f"{run_name} — ORT cumulative energy",
            ci_lo_key="cum_mj_ci95_lo",
            ci_hi_key="cum_mj_ci95_hi",
            trial_totals=list(data.get("energy_mj") or []),
        )
        if energy_fig is not None:
            if any(float(n.get("cum_mj_mean") or 0) for n in ort_nodes):
                png_path = plots_dir / "lop7_ort_profiler_index_vs_energy.png"
                energy_fig.savefig(png_path, dpi=150)
                media_images["plots/ort_cumulative_energy"] = wandb.Image(
                    str(png_path), caption="ORT cumulative energy vs layer index"
                )
            plt.close(energy_fig)

    if media_images:
        wandb.log(media_images)
        run.summary.update(media_images)
        print(f"[visualize] Histogram PNGs: {hist_dir}", flush=True)
        print(f"[visualize] Series PNGs: {series_dir}", flush=True)


def log_result(
    result_path: str | Path,
    project: str = "geyikbench",
    wandb_dir: str | Path | None = None,
) -> None:
    """Load a result.json and store summary metrics + plots in W&B.

    Scalars go to ``run.summary`` only (no ``wandb.log`` of scalars), so W&B
    does not create a chart panel per metric. Per-sample distributions are
    logged as matplotlib histogram and sample-vs-metric images.

    W&B local run files are written into *wandb_dir* (defaults to the same
    directory as result.json so they live alongside the Hydra outputs).

    Args:
        result_path: Path to result.json (or its parent directory).
        project:     W&B project name.
        wandb_dir:   Directory where the ``wandb/`` subfolder will be created.
                     Defaults to ``result_path.parent``.
    """
    # Load .env only when available (not present inside Docker; vars are injected via --env-file)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    result_path = Path(result_path)
    if result_path.is_dir():
        result_path = result_path / "result.json"

    output_dir = result_path.parent  # e.g. outputs/2026-07-29_09-04-41_lop7/

    if wandb_dir is None:
        wandb_dir = output_dir
    wandb_dir = Path(wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    with open(result_path) as f:
        data = json.load(f)

    experiment = data.get("experiment", {})
    run_name = experiment.get("name", output_dir.parent.name)
    timestamp = output_dir.name  # e.g. "2026-07-29_09-04-41"

    config = {
        "experiment": experiment,
        "model": data["model"],
        "device_index": data["device_index"],
        "device_name": data["device_name"],
        "backend": data["backend"],
        "providers": data["providers"],
        "warmup_s": data["warmup_s"],
        "runs_s": data["runs_s"],
        "warmup_iters": data["warmup_iters"],
        "runs_iters": data["runs_iters"],
        "result_path": str(result_path.resolve()),
    }

    run = wandb.init(
        project=project,
        name=f"{run_name}_{timestamp}",
        config=config,
        dir=str(wandb_dir),
    )

    populate_wandb_run(run, data, output_dir=output_dir, run_name=run_name)

    wandb.finish()
    print(f"[visualize] W&B run '{run.name}' logged to project '{project}'.", flush=True)
    print(f"[visualize] Local W&B files: {wandb_dir / 'wandb'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log benchmark result.json to W&B")
    parser.add_argument(
        "result",
        type=str,
        help="Path to result.json or its parent directory",
    )
    parser.add_argument("--project", type=str, default="geyikbench", help="W&B project name")
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default=None,
        help="Directory for local W&B files (default: same dir as result.json)",
    )
    args = parser.parse_args()
    log_result(args.result, project=args.project, wandb_dir=args.wandb_dir)


if __name__ == "__main__":
    main()
