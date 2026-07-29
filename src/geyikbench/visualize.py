"""Log benchmark results to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import wandb


def log_result(
    result_path: str | Path,
    project: str = "geyikbench",
    wandb_dir: str | Path | None = None,
) -> None:
    """Load a result.json and log summary metrics + per-sample tables to W&B.

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

    # Derive std for metrics that only have mean stored at the top level
    powers = data["power_w"]
    temps = data["temperature_c"]
    freqs = data["frequency_mhz"]

    summary = {
        "latency_ms/mean": data["latency_ms_mean"],
        "latency_ms/std": data["latency_ms_std"],
        "latency_ms/p50": data["latency_ms_p50"],
        "latency_ms/p95": data["latency_ms_p95"],
        "energy_mj/mean": data["energy_mj_mean"],
        "energy_mj/std": data["energy_mj_std"],
        "energy_mj/total": data["energy_mj_total"],
        "power_w/mean": data["power_w_mean"],
        "power_w/std": statistics.pstdev(powers) if len(powers) > 1 else 0.0,
        "temperature_c/mean": data["temperature_c_mean"],
        "temperature_c/std": statistics.pstdev(temps) if len(temps) > 1 else 0.0,
        "frequency_mhz/mean": data["frequency_mhz_mean"],
        "frequency_mhz/std": statistics.pstdev(freqs) if len(freqs) > 1 else 0.0,
        "throughput_inf_per_s": data["throughput_inf_per_s"],
    }

    run.summary.update(summary)
    wandb.log(summary)

    # Per-sample table
    n = len(data["latency_ms"])
    table = wandb.Table(
        columns=["step", "latency_ms", "energy_mj", "power_w", "temperature_c", "frequency_mhz"],
    )
    for i in range(n):
        table.add_data(
            i,
            data["latency_ms"][i],
            data["energy_mj"][i],
            data["power_w"][i],
            data["temperature_c"][i],
            data["frequency_mhz"][i],
        )
    wandb.log({"samples": table})

    # Histograms
    non_zero_energy = [e for e in data["energy_mj"] if e > 0]
    hist_payload: dict = {
        "latency_ms/histogram": wandb.Histogram(data["latency_ms"]),
        "power_w/histogram": wandb.Histogram(data["power_w"]),
        "temperature_c/histogram": wandb.Histogram(data["temperature_c"]),
        "frequency_mhz/histogram": wandb.Histogram(data["frequency_mhz"]),
    }
    if non_zero_energy:
        hist_payload["energy_mj/histogram"] = wandb.Histogram(non_zero_energy)
    wandb.log(hist_payload)

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
