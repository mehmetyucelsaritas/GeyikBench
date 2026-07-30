"""W&B Sweep agent: one (warmup_s, runs_s) config with repeats → latency CV."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np
import wandb

from geyikbench.benchmark import benchmark_onnx
from geyikbench.visualize import populate_wandb_run


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = statistics.fmean(values)
    if mean == 0.0:
        return float("nan")
    return statistics.pstdev(values) / abs(mean)


def _normalize_sweep_id(sweep_id: object | None) -> str:
    """Keep only the sweep token from ``entity/project/id`` or a bare id."""
    if not sweep_id:
        return "no_sweep"
    return str(sweep_id).rstrip("/").split("/")[-1]


def _sweep_id_for_run(run: object) -> str:
    """Prefer the active W&B sweep id so local outputs group by sweep."""
    sweep_id = getattr(run, "sweep_id", None) or os.environ.get("WANDB_SWEEP_ID")
    return _normalize_sweep_id(sweep_id)


def _sweep_root() -> Path:
    """Directory for one sweep: ``outputs/<date>_timing_sweep_<sweep_id>``."""
    import re
    from datetime import datetime

    sweep_id = _normalize_sweep_id(os.environ.get("WANDB_SWEEP_ID"))
    wandb_dir_env = os.environ.get("WANDB_DIR", "").strip()
    if wandb_dir_env:
        root = Path(wandb_dir_env)
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    label = "timing_sweep"
    parent = Path("outputs")
    parent.mkdir(parents=True, exist_ok=True)
    dated_re = re.compile(
        rf"^\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}"
        rf"(?:_{re.escape(label)})?_{re.escape(sweep_id)}$"
    )
    dated = sorted(
        (p for p in parent.iterdir() if p.is_dir() and dated_re.match(p.name)),
        key=lambda p: p.name,
    )
    if dated:
        return dated[-1].resolve()
    legacy_parent = parent / "wandb_timing_sweep"
    if legacy_parent.is_dir():
        legacy_dated = sorted(legacy_parent.glob(f"*_{sweep_id}"))
        if legacy_dated:
            return legacy_dated[-1].resolve()
        legacy_bare = legacy_parent / sweep_id
        if legacy_bare.is_dir():
            return legacy_bare.resolve()
    bare = parent / sweep_id
    if bare.is_dir():
        return bare.resolve()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = parent / f"{stamp}_{label}_{sweep_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _run_name_from_sweep_params() -> str | None:
    """Build ``warmupX_runsY`` from the agent-written sweep param YAML, if present."""
    param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH", "").strip()
    if not param_path or not Path(param_path).is_file():
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(param_path)

        def _value(key: str) -> float:
            raw = OmegaConf.select(cfg, f"{key}.value")
            if raw is None:
                raw = OmegaConf.select(cfg, key)
            return float(raw)

        return f"warmup{_value('warmup_s'):g}_runs{_value('runs_s'):g}"
    except Exception:
        return None


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    sweep_root = _sweep_root()
    # Agent sets WANDB_RUN_ID before launching this script — use it so wandb.init
    # and our artifacts share one folder (no parallel warmup*_ + wandb/run-* trees).
    run_id = os.environ.get("WANDB_RUN_ID", "").strip() or wandb.util.generate_id()
    run_name = _run_name_from_sweep_params() or f"run_{run_id}"
    out_dir = sweep_root / f"{run_name}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(dir=str(out_dir), id=run_id, name=run_name)
    cfg = run.config
    sweep_id = _sweep_id_for_run(run)

    warmup_s = float(cfg.warmup_s)
    runs_s = float(cfg.runs_s)
    repeats = int(cfg.get("repeats", 3))
    device = int(cfg.get("device", 0))
    batch_size = int(cfg.get("batch_size", 1))
    seed = int(cfg.get("seed", 42))
    model = Path(str(cfg.get("model", "models/LOP7/nnlf_lop7_model_float.onnx")))
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model.resolve()}")

    # Prefer config-derived name once init has loaded sweep params.
    run_name = f"warmup{warmup_s:g}_runs{runs_s:g}"
    run.name = run_name

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    run.config.update(
        {
            "model_resolved": str(model.resolve()),
            "sweep_id": sweep_id,
            "output_dir": str(out_dir.resolve()),
        },
        allow_val_change=True,
    )

    trial_payloads: list[dict] = []
    latency_means: list[float] = []
    energy_means: list[float] = []
    power_means: list[float] = []
    temp_means: list[float] = []
    latency_stds: list[float] = []
    energy_stds: list[float] = []
    within_cvs: list[float] = []

    for rep in range(repeats):
        np.random.seed(seed + rep)
        result = benchmark_onnx(
            model_path=model,
            device_index=device,
            warmup_s=warmup_s,
            runs_s=runs_s,
            warmup_iters=None,
            runs_iters=None,
            batch_size=batch_size,
        )
        payload = result.to_dict()
        trial_path = trials_dir / f"rep_{rep}.json"
        trial_path.write_text(json.dumps(payload, indent=2) + "\n")
        trial_payloads.append(payload)

        within_cv = (
            result.latency_ms_std / abs(result.latency_ms_mean)
            if result.latency_ms_mean != 0.0
            else float("nan")
        )
        latency_means.append(result.latency_ms_mean)
        energy_means.append(result.energy_mj_mean)
        power_means.append(result.power_w_mean)
        temp_means.append(result.temperature_c_mean)
        latency_stds.append(result.latency_ms_std)
        energy_stds.append(result.energy_mj_std)
        within_cvs.append(within_cv)

        print(
            f"[rep={rep} gpu={result.device_index}] "
            f"latency_ms_mean={result.latency_ms_mean:.4f} "
            f"std={result.latency_ms_std:.4f}",
            flush=True,
        )

    # Use the last trial's full per-sample series for histograms/series (same as
    # individual benchmark runs). Scalar table metrics average across repeats.
    primary = dict(trial_payloads[-1])
    primary["latency_ms_mean"] = statistics.fmean(latency_means)
    primary["latency_ms_std"] = statistics.fmean(latency_stds)
    primary["latency_ms_p50"] = statistics.fmean(
        [float(t["latency_ms_p50"]) for t in trial_payloads]
    )
    primary["latency_ms_p95"] = statistics.fmean(
        [float(t["latency_ms_p95"]) for t in trial_payloads]
    )
    primary["energy_mj_mean"] = statistics.fmean(energy_means)
    primary["energy_mj_std"] = statistics.fmean(energy_stds)
    primary["energy_mj_total"] = statistics.fmean(
        [float(t["energy_mj_total"]) for t in trial_payloads]
    )
    primary["power_w_mean"] = statistics.fmean(power_means)
    primary["temperature_c_mean"] = statistics.fmean(temp_means)
    primary["frequency_mhz_mean"] = statistics.fmean(
        [float(t["frequency_mhz_mean"]) for t in trial_payloads]
    )
    primary["frequency_mem_mhz_mean"] = statistics.fmean(
        [float(t["frequency_mem_mhz_mean"]) for t in trial_payloads]
    )
    primary["frequency_graphics_mhz_mean"] = statistics.fmean(
        [float(t["frequency_graphics_mhz_mean"]) for t in trial_payloads]
    )
    primary["frequency_video_mhz_mean"] = statistics.fmean(
        [float(t["frequency_video_mhz_mean"]) for t in trial_payloads]
    )
    primary["throughput_inf_per_s"] = (
        1000.0 / primary["latency_ms_mean"] if primary["latency_ms_mean"] > 0 else float("nan")
    )

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(primary, indent=2) + "\n")

    latency_cv = _cv(latency_means)
    energy_cv = _cv(energy_means)
    extra_summary = {
        # Sweep objective (also declared in configs/wandb/sweep_timing.yaml).
        "latency_ms_mean_cv": latency_cv,
        "energy_mj_mean_cv": energy_cv,
        "latency_ms_mean_of_means": statistics.fmean(latency_means),
        "latency_ms_std_of_means": (
            statistics.pstdev(latency_means) if len(latency_means) > 1 else 0.0
        ),
        "energy_mj_mean_of_means": statistics.fmean(energy_means),
        "within_run_latency_cv_mean": (
            statistics.fmean(within_cvs) if within_cvs else float("nan")
        ),
        "repeats_done": repeats,
    }

    populate_wandb_run(
        run,
        primary,
        output_dir=out_dir,
        run_name=run_name,
        extra_summary=extra_summary,
    )

    print(
        f"Done warmup_s={warmup_s:g} runs_s={runs_s:g} "
        f"latency_ms/mean={primary['latency_ms_mean']:.4f} "
        f"latency_ms_mean_cv={latency_cv:.6f}",
        flush=True,
    )
    print(f"Wrote {result_path}", flush=True)
    print(f"Run directory: {out_dir}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
