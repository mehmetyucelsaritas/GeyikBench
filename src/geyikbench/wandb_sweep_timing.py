"""W&B Sweep agent: profile ONNX models for latency/energy (optional ORT nodes)."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np
import wandb

from geyikbench.benchmark import benchmark_onnx
from geyikbench.visualize import populate_wandb_run


def _profile_ort_coupled(
    *,
    model: Path,
    device: int,
    warmup_s: float,
    runs_s: float,
    batch_size: int,
    n_trials: int,
    settle_iters: int,
    out_dir: Path,
) -> dict:
    """Run ORT profiler trials then NVML power trials; return merged payload."""
    from geyikbench.ort_profiler import merge_ort_runtime_nvml_power, profile_onnx

    if n_trials < 1:
        raise ValueError(f"trials must be >= 1, got {n_trials}")

    profile_prefix = out_dir / "ort_profile"
    ort_trials = []
    print(f"[INFO] Pass 1/2: ORT profiler (runtime) ×{n_trials} trial(s)", flush=True)
    for t in range(n_trials):
        prefix = profile_prefix if n_trials == 1 else Path(f"{profile_prefix}_trial{t:02d}")
        print(f"[INFO] ORT trial {t + 1}/{n_trials}", flush=True)
        ort_trials.append(
            profile_onnx(
                model_path=model,
                device_index=device,
                warmup_s=warmup_s,
                runs_s=runs_s,
                warmup_iters=None,
                runs_iters=None,
                batch_size=batch_size,
                profile_file_prefix=prefix,
                settle_iters=settle_iters,
            )
        )

    nvml_trials = []
    print(f"[INFO] Pass 2/2: benchmark_onnx (NVML power) ×{n_trials} trial(s)", flush=True)
    for t in range(n_trials):
        print(f"[INFO] power trial {t + 1}/{n_trials}", flush=True)
        nvml_trials.append(
            benchmark_onnx(
                model_path=model,
                device_index=device,
                warmup_s=warmup_s,
                runs_s=runs_s,
                warmup_iters=None,
                runs_iters=None,
                batch_size=batch_size,
            )
        )

    payload = merge_ort_runtime_nvml_power(
        nvml_trials[-1],
        ort_trials[-1],
        nvml_results=nvml_trials,
        ort_trials=ort_trials,
    )
    print(
        f"[INFO] ORT profiler: trials={n_trials} "
        f"n_nodes={payload['ort_profiler']['n_nodes']} "
        f"latency_ms_mean={payload['latency_ms_mean']:.4f} "
        f"power_w_mean={payload['power_w_mean']:.4f}",
        flush=True,
    )
    return payload


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


def _write_run_outputs(
    *,
    run: object,
    primary: dict,
    out_dir: Path,
    run_name: str,
    latency_means: list[float],
    energy_means: list[float],
    within_cvs: list[float],
    repeats_done: int,
    trials: int,
    use_ort: bool,
) -> Path:
    """Persist result.json / ort_nodes.json and push summary + plots to W&B."""
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(primary, indent=2) + "\n")

    ort_block = primary.get("ort_profiler")
    if isinstance(ort_block, dict) and ort_block.get("nodes"):
        nodes_path = out_dir / "ort_nodes.json"
        nodes_path.write_text(json.dumps(ort_block["nodes"], indent=2) + "\n")
        print(f"Wrote {nodes_path}", flush=True)

    latency_cv = _cv(latency_means)
    energy_cv = _cv(energy_means)
    extra_summary = {
        "latency_ms_mean_cv": latency_cv,
        "energy_mj_mean_cv": energy_cv,
        "latency_ms_mean_of_means": (
            statistics.fmean(latency_means) if latency_means else primary["latency_ms_mean"]
        ),
        "latency_ms_std_of_means": (
            statistics.pstdev(latency_means) if len(latency_means) > 1 else 0.0
        ),
        "energy_mj_mean_of_means": (
            statistics.fmean(energy_means) if energy_means else primary["energy_mj_mean"]
        ),
        "within_run_latency_cv_mean": (
            statistics.fmean(within_cvs) if within_cvs else float("nan")
        ),
        "repeats_done": repeats_done,
        "trials_done": trials if use_ort else repeats_done,
        "use_ort_profiler": use_ort,
        "n_nodes": int((ort_block or {}).get("n_nodes") or 0) if use_ort else 0,
    }

    populate_wandb_run(
        run,
        primary,
        output_dir=out_dir,
        run_name=run_name,
        extra_summary=extra_summary,
    )

    print(
        f"Done warmup_s={float(primary.get('warmup_s', float('nan'))):g} "
        f"runs_s={float(primary.get('runs_s', float('nan'))):g} "
        f"latency_ms/mean={primary['latency_ms_mean']:.4f} "
        f"latency_ms_mean_cv={latency_cv:.6f}",
        flush=True,
    )
    print(f"Wrote {result_path}", flush=True)
    print(f"Run directory: {out_dir}", flush=True)
    return result_path


def _resolve_run_id(run_dir: Path) -> str:
    """Infer the W&B run id from ``wandb/run-*`` folders or the directory name."""
    wandb_dir = run_dir / "wandb"
    if wandb_dir.is_dir():
        runs = sorted(wandb_dir.glob("run-*-*"))
        if runs:
            # run-YYYYMMDD_HHMMSS-<id>
            return runs[-1].name.rsplit("-", 1)[-1]
    # Fallback: trailing token after the last underscore (e.g. warmup10_runs10_fzfw5b7m).
    return run_dir.name.rsplit("_", 1)[-1]


def _load_wandb_config_values(run_dir: Path) -> dict:
    """Load hyperparameter values from a previous run's ``config.yaml`` if present."""
    import yaml

    configs = sorted((run_dir / "wandb").rglob("config.yaml")) if (run_dir / "wandb").is_dir() else []
    if not configs:
        return {}
    raw = yaml.safe_load(configs[-1].read_text()) or {}
    out: dict = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and "value" in value:
            out[key] = value["value"]
        else:
            out[key] = value
    return out


def rerun_existing(
    run_dir: str | Path,
    *,
    device: int | None = None,
    project: str | None = None,
    trials: int | None = None,
    settle_iters: int | None = None,
) -> Path:
    """Re-profile into an existing sweep run dir and resume the same W&B run.

    Overwrites ``result.json``, ``ort_nodes.json``, and local plot artifacts, then
    updates the resumed W&B run summary/media in place (same run id).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    prev = {}
    result_path = run_dir / "result.json"
    if result_path.is_file():
        prev = json.loads(result_path.read_text())
    cfg = _load_wandb_config_values(run_dir)

    model = Path(str(cfg.get("model") or prev.get("model") or ""))
    if not model.is_file():
        # Paths in result.json may be relative to the repo root.
        candidate = Path.cwd() / model
        if candidate.is_file():
            model = candidate
    if not model.is_file():
        raise FileNotFoundError(f"Model not found for rerun: {model}")

    warmup_s = float(cfg.get("warmup_s", prev.get("warmup_s", 10)))
    runs_s = float(cfg.get("runs_s", prev.get("runs_s", 10)))
    batch_size = int(cfg.get("batch_size", 1) or 1)
    seed = int(cfg.get("seed", 42))
    use_ort = bool(cfg.get("use_ort_profiler", prev.get("latency_source") == "ort_profiler"))
    n_trials = int(trials if trials is not None else cfg.get("trials", prev.get("trials", 1)))
    n_settle = int(
        settle_iters if settle_iters is not None else cfg.get("settle_iters", 3)
    )
    # Prefer explicit CLI device; else physical GPU recorded in result.json; else sweep param.
    if device is not None:
        device_index = int(device)
    elif prev.get("device_index") is not None:
        device_index = int(prev["device_index"])
    else:
        device_index = int(cfg.get("device", 0))

    project_name = str(project or cfg.get("project") or "geyikbench")
    run_id = _resolve_run_id(run_dir)
    run_name = f"{model.stem}_warmup{warmup_s:g}_runs{runs_s:g}"

    print(
        f"[rerun] dir={run_dir} id={run_id} model={model} "
        f"device={device_index} trials={n_trials} use_ort={use_ort}",
        flush=True,
    )

    np.random.seed(seed)
    run = wandb.init(
        project=project_name,
        id=run_id,
        name=run_name,
        resume="must",
        dir=str(run_dir),
    )
    run.name = run_name
    run.config.update(
        {
            "model": str(model),
            "model_resolved": str(model.resolve()),
            "warmup_s": warmup_s,
            "runs_s": runs_s,
            "trials": n_trials,
            "use_ort_profiler": use_ort,
            "settle_iters": n_settle,
            "device": device_index,
            "batch_size": batch_size,
            "seed": seed,
            "output_dir": str(run_dir),
            "rerun": True,
        },
        allow_val_change=True,
    )

    if not use_ort:
        raise ValueError("rerun_existing currently supports use_ort_profiler runs only")

    primary = _profile_ort_coupled(
        model=model,
        device=device_index,
        warmup_s=warmup_s,
        runs_s=runs_s,
        batch_size=batch_size,
        n_trials=n_trials,
        settle_iters=n_settle,
        out_dir=run_dir,
    )
    latency_means = [float(x) for x in (primary.get("latency_ms") or [])]
    energy_means = [float(x) for x in (primary.get("energy_mj") or [])]
    _write_run_outputs(
        run=run,
        primary=primary,
        out_dir=run_dir,
        run_name=run_name,
        latency_means=latency_means,
        energy_means=energy_means,
        within_cvs=[],
        repeats_done=1,
        trials=n_trials,
        use_ort=True,
    )
    wandb.finish()
    return run_dir


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
    trials = int(cfg.get("trials", 1))
    use_ort = bool(cfg.get("use_ort_profiler", False))
    settle_iters = int(cfg.get("settle_iters", 3))
    device = int(cfg.get("device", 0))
    batch_size = int(cfg.get("batch_size", 1))
    seed = int(cfg.get("seed", 42))
    model = Path(str(cfg.get("model", "models/LOP7/nnlf_lop7_model_float.onnx")))
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model.resolve()}")

    # Prefer config-derived name once init has loaded sweep params.
    # Include model stem so multi-model grids are distinct.
    run_name = f"{model.stem}_warmup{warmup_s:g}_runs{runs_s:g}"
    run.name = run_name

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    run.config.update(
        {
            "model_resolved": str(model.resolve()),
            "sweep_id": sweep_id,
            "output_dir": str(out_dir.resolve()),
            "use_ort_profiler": use_ort,
            "trials": trials,
        },
        allow_val_change=True,
    )

    np.random.seed(seed)

    if use_ort:
        # One merged result with across-trial means/CI (same as invoke benchmark).
        primary = _profile_ort_coupled(
            model=model,
            device=device,
            warmup_s=warmup_s,
            runs_s=runs_s,
            batch_size=batch_size,
            n_trials=trials,
            settle_iters=settle_iters,
            out_dir=out_dir,
        )
        latency_means = [float(x) for x in (primary.get("latency_ms") or [])]
        energy_means = [float(x) for x in (primary.get("energy_mj") or [])]
        within_cvs = []
        repeats_done = 1
    else:
        trial_payloads: list[dict] = []
        latency_means = []
        energy_means = []
        power_means: list[float] = []
        temp_means: list[float] = []
        latency_stds: list[float] = []
        energy_stds: list[float] = []
        within_cvs = []

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

        # Use the last trial's full per-sample series for histograms/series.
        # Scalar table metrics average across repeats.
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
        repeats_done = repeats

    _write_run_outputs(
        run=run,
        primary=primary,
        out_dir=out_dir,
        run_name=run_name,
        latency_means=latency_means,
        energy_means=energy_means,
        within_cvs=within_cvs,
        repeats_done=repeats_done,
        trials=trials,
        use_ort=use_ort,
    )
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="W&B timing sweep agent / rerun")
    parser.add_argument(
        "--rerun-dir",
        type=str,
        default="",
        help="Existing sweep run directory to overwrite (resume same W&B run id)",
    )
    parser.add_argument("--device", type=int, default=None, help="GPU index override for --rerun-dir")
    parser.add_argument("--project", type=str, default=None, help="W&B project override for --rerun-dir")
    parser.add_argument("--trials", type=int, default=None, help="ORT/NVML trials override for --rerun-dir")
    parser.add_argument(
        "--settle-iters",
        type=int,
        default=None,
        help="ORT settle_iters override for --rerun-dir",
    )
    args, _unknown = parser.parse_known_args()
    if args.rerun_dir:
        rerun_existing(
            args.rerun_dir,
            device=args.device,
            project=args.project,
            trials=args.trials,
            settle_iters=args.settle_iters,
        )
    else:
        main()
