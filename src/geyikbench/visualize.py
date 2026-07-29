"""Log benchmark results to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


# Keep in sync with geyikbench.benchmark._METRIC_SKIP_S / _steady_start_index.
_METRIC_SKIP_S = 1.0


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

    output_dir = result_path.parent  # e.g. outputs/lop7_baseline/2026-07-29_09-04-41/

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

    # Derive std for metrics that only have mean stored at the top level.
    # Use `.get(...)` so older `result.json` files (without the new clock domains)
    # can still be visualized. Match benchmark.py: skip first 1s of measured time.
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

    summary = {
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

    # Summary-only: do not wandb.log() scalars — that creates one empty
    # single-point chart panel per metric. Summary metrics still show up in
    # the run overview and the project runs table for cross-run comparison.
    run.summary.update(summary)

    # Per-sample table (one media panel; useful for inspecting distributions).
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
        freq_sm_i = freqs_sm[i] if i < len(freqs_sm) else float("nan")
        freq_mem_i = freqs_mem[i] if i < len(freqs_mem) else float("nan")
        freq_graphics_i = freqs_graphics[i] if i < len(freqs_graphics) else float("nan")
        freq_video_i = freqs_video[i] if i < len(freqs_video) else float("nan")
        table.add_data(
            i,
            latencies[i],
            energies[i] if i < len(energies) else float("nan"),
            powers[i] if i < len(powers) else float("nan"),
            temps[i] if i < len(temps) else float("nan"),
            freq_sm_i,
            freq_mem_i,
            freq_graphics_i,
            freq_video_i,
        )
    run.summary["samples"] = table

    # Matplotlib plots as images (not W&B Histogram objects / scalar charts).
    # energy_mj is derived as power_w * latency_ms (W*ms == mJ), not NVML energy.
    # (key, series_for_sample_plot, series_for_histogram, axis_label)
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

    if media_images:
        # One media log step — images appear under Media, not as scalar charts.
        wandb.log(media_images)
        run.summary.update(media_images)

    wandb.finish()
    print(f"[visualize] W&B run '{run.name}' logged to project '{project}'.", flush=True)
    print(f"[visualize] Local W&B files: {wandb_dir / 'wandb'}", flush=True)
    if media_images:
        print(f"[visualize] Histogram PNGs: {hist_dir}", flush=True)
        print(f"[visualize] Series PNGs: {series_dir}", flush=True)


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
