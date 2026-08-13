"""ONNX Runtime native profiler helpers for per-node runtime labels.

Two separate passes when used from ``benchmark.py`` with
``use_ort_profiler=true``:

1. ORT ``enable_profiling`` → runtime labels (``*_kernel_time``)
2. ``benchmark_onnx`` (no profiler) → NVML power labels

With ``benchmark.trials>1``, all ORT trials run back-to-back, then all NVML
power trials; labels are means across trials.

Energy is ``ort_latency_ms * power_w``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geyikbench.benchmark import (
    _create_ort_session,
    _cuda_synchronize,
    _make_onnx_feeds,
    _percentile,
    _run_until,
)


@dataclass
class OrtNodeTiming:
    """Mean timing and energy for one graph node across profiled inferences.

    Energy fields are filled in ``merge_ort_runtime_nvml_power`` as
    ``dur_ms * power_w`` (mJ). Zero until power is known.
    """

    index: int
    name: str
    op_name: str
    dur_ms_mean: float
    dur_ms_std: float
    cum_ms_mean: float
    energy_mj_mean: float = 0.0
    energy_mj_std: float = 0.0
    cum_mj_mean: float = 0.0
    cum_ms_ci95_lo: float = float("nan")
    cum_ms_ci95_hi: float = float("nan")
    cum_mj_ci95_lo: float = float("nan")
    cum_mj_ci95_hi: float = float("nan")


@dataclass
class OrtProfileResult:
    """Aggregated ORT profiler timings for one ONNX model."""

    model: str
    providers: list[str]
    warmup_s: float | None
    runs_s: float | None
    warmup_iters: int
    runs_iters: int
    settle_iters: int
    n_nodes: int
    profile_path: str
    # Per-inference totals (sum of kernel_time over nodes), ms.
    latency_ms: list[float]
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    nodes: list[OrtNodeTiming]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        payload = asdict(self)
        payload["nodes"] = [asdict(n) for n in self.nodes]
        return payload


def _is_kernel_node(event: dict[str, Any]) -> bool:
    """True for op kernel timings (exclude fence_before / fence_after helpers)."""
    name = str(event.get("name") or "")
    return name.endswith("_kernel_time")


def parse_kernel_runs(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split chrome-trace events into per-inference lists of kernel Node events.

    Uses a two-pointer scan over sorted kernel events and ``model_run`` session
    windows so parsing stays linear in the number of profile events.
    """
    kernels = sorted(
        (e for e in events if e.get("cat") == "Node" and "dur" in e and _is_kernel_node(e)),
        key=lambda e: e["ts"],
    )
    runs_meta = sorted(
        (e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"),
        key=lambda e: e["ts"],
    )
    runs: list[list[dict[str, Any]]] = []
    ki = 0
    n_k = len(kernels)
    for run in runs_meta:
        t0 = run["ts"]
        t1 = t0 + run["dur"]
        while ki < n_k and kernels[ki]["ts"] < t0:
            ki += 1
        start = ki
        while ki < n_k and kernels[ki]["ts"] < t1:
            ki += 1
        if ki > start:
            runs.append(kernels[start:ki])
    return runs


def aggregate_kernel_runs(
    runs: list[list[dict[str, Any]]],
    *,
    settle_iters: int = 0,
) -> tuple[list[float], list[OrtNodeTiming]]:
    """Aggregate per-run kernel lists into total latencies and per-node means.

    Args:
        runs: One list of kernel Node events per ``session.run``.
        settle_iters: Drop this many leading runs (profiled-session settle).

    Returns:
        ``(per_run_total_ms, node_timings)`` where node timings use the modal
        node-count runs only.
    """
    if settle_iters > 0 and len(runs) > settle_iters:
        runs = runs[settle_iters:]
    if not runs:
        raise RuntimeError("No ORT profiled runs to aggregate")

    counts = Counter(len(r) for r in runs)
    modal_n, _ = counts.most_common(1)[0]
    runs = [r for r in runs if len(r) == modal_n]
    if not runs:
        raise RuntimeError("No ORT profiled runs with a consistent node count")

    durs_us = np.asarray([[float(e["dur"]) for e in run] for run in runs], dtype=np.float64)
    totals_ms = (durs_us.sum(axis=1) / 1000.0).tolist()
    mean_ms = durs_us.mean(axis=0) / 1000.0
    std_ms = durs_us.std(axis=0, ddof=1) / 1000.0 if len(runs) > 1 else np.zeros_like(mean_ms)
    cum_ms = np.cumsum(mean_ms)
    names = [str(e["name"]) for e in runs[0]]
    op_types = [str((e.get("args") or {}).get("op_name") or "") for e in runs[0]]
    nodes = [
        OrtNodeTiming(
            index=i,
            name=names[i],
            op_name=op_types[i],
            dur_ms_mean=float(mean_ms[i]),
            dur_ms_std=float(std_ms[i]),
            cum_ms_mean=float(cum_ms[i]),
        )
        for i in range(modal_n)
    ]
    return totals_ms, nodes


def profile_onnx(
    model_path: Path,
    device_index: int = 0,
    warmup_s: float | None = 5.0,
    runs_s: float | None = 5.0,
    warmup_iters: int | None = None,
    runs_iters: int | None = None,
    batch_size: int = 1,
    *,
    profile_file_prefix: str | Path,
    settle_iters: int = 3,
) -> OrtProfileResult:
    """Warm up, then ORT-profile a measurement window (runtime only, no NVML).

    Warmup uses a non-profiled session. The profiled session runs
    ``settle_iters`` untimed inferences, then measures until ``runs_s`` /
    ``runs_iters``. The chrome-trace is parsed into node timings and then
    deleted; keep ``result.json`` / ``ort_nodes.json`` for labels.
    """
    model_path = Path(model_path)
    profile_file_prefix = Path(profile_file_prefix)
    profile_file_prefix.parent.mkdir(parents=True, exist_ok=True)

    warm_sess = _create_ort_session(model_path, device_index)
    feeds = _make_onnx_feeds(warm_sess, batch_size)
    outs = [o.name for o in warm_sess.get_outputs()]
    warmup_iters_done = _run_until(
        lambda: warm_sess.run(outs, feeds),
        duration_s=warmup_s,
        max_iters=warmup_iters,
    )
    _cuda_synchronize(device_index)
    del warm_sess

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = str(profile_file_prefix)
    providers: list = [
        ("CUDAExecutionProvider", {"device_id": device_index}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    feeds = _make_onnx_feeds(sess, batch_size)
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(max(0, settle_iters)):
        sess.run(outs, feeds)
    _cuda_synchronize(device_index)
    measure_iters = _run_until(
        lambda: sess.run(outs, feeds),
        duration_s=runs_s,
        max_iters=runs_iters,
    )
    _cuda_synchronize(device_index)
    profile_path = Path(sess.end_profiling())
    providers_active = list(sess.get_providers())
    del sess

    try:
        events = json.loads(profile_path.read_text())
        runs = parse_kernel_runs(events)
        del events
        if measure_iters > 0 and len(runs) > measure_iters:
            runs = runs[-measure_iters:]
            settle_applied = 0
        else:
            settle_applied = settle_iters
        totals_ms, nodes = aggregate_kernel_runs(runs, settle_iters=settle_applied)
    finally:
        profile_path.unlink(missing_ok=True)

    return OrtProfileResult(
        model=str(model_path),
        providers=providers_active,
        warmup_s=warmup_s,
        runs_s=runs_s,
        warmup_iters=warmup_iters_done,
        runs_iters=measure_iters,
        settle_iters=settle_iters,
        n_nodes=len(nodes),
        profile_path="",
        latency_ms=totals_ms,
        latency_ms_mean=statistics.fmean(totals_ms) if totals_ms else float("nan"),
        latency_ms_std=statistics.pstdev(totals_ms) if len(totals_ms) > 1 else 0.0,
        latency_ms_p50=_percentile(totals_ms, 50),
        latency_ms_p95=_percentile(totals_ms, 95),
        nodes=nodes,
    )


def _t_crit_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for df = n - 1."""
    table = {
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
    if n < 2:
        return float("nan")
    return table.get(n, 1.96)


def _attach_node_energy(nodes: list[OrtNodeTiming], power_w: float) -> None:
    """Set per-node and cumulative energy from NVML mean power (mJ = ms × W)."""
    if not (power_w == power_w and power_w > 0):
        return
    for node in nodes:
        node.energy_mj_mean = float(node.dur_ms_mean) * power_w
        node.energy_mj_std = float(node.dur_ms_std) * power_w
        node.cum_mj_mean = float(node.cum_ms_mean) * power_w
        if node.cum_ms_ci95_lo == node.cum_ms_ci95_lo:
            node.cum_mj_ci95_lo = float(node.cum_ms_ci95_lo) * power_w
            node.cum_mj_ci95_hi = float(node.cum_ms_ci95_hi) * power_w


def _mean_or_nan(values: list[float]) -> float:
    """Arithmetic mean, or NaN when empty."""
    return statistics.fmean(values) if values else float("nan")


def _std_or_zero(values: list[float]) -> float:
    """Population stddev across values, or 0 when fewer than 2 samples."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_ort_trials(orts: list[OrtProfileResult]) -> OrtProfileResult:
    """Average per-node durations and latency summaries across ORT trials.

    ``latency_ms`` becomes the list of per-trial mean latencies (one value per
    trial). ``latency_ms_std`` is the across-trial std of those means.
    """
    if not orts:
        raise ValueError("aggregate_ort_trials requires at least one OrtProfileResult")
    if len(orts) == 1:
        return orts[0]

    n_nodes = Counter(o.n_nodes for o in orts).most_common(1)[0][0]
    kept = [o for o in orts if o.n_nodes == n_nodes]
    if not kept:
        raise ValueError("No ORT trials share a common node count")

    ref = kept[0]
    trial_means = [float(o.latency_ms_mean) for o in kept]
    dur_stack = np.stack(
        [np.array([n.dur_ms_mean for n in o.nodes], dtype=np.float64) for o in kept],
        axis=0,
    )
    std_stack = np.stack(
        [np.array([n.dur_ms_std for n in o.nodes], dtype=np.float64) for o in kept],
        axis=0,
    )
    mean_ms = dur_stack.mean(axis=0)
    # Prefer across-trial std of means; fall back to mean within-trial std.
    across_std = dur_stack.std(axis=0, ddof=0) if len(kept) > 1 else std_stack[0]
    cum_stack = np.cumsum(dur_stack, axis=1)
    n_trials = int(cum_stack.shape[0])
    cum_ms = cum_stack.mean(axis=0)
    if n_trials >= 2:
        sem = cum_stack.std(axis=0, ddof=1) / float(n_trials) ** 0.5
        t_crit = _t_crit_95(n_trials)
        cum_lo = cum_ms - t_crit * sem
        cum_hi = cum_ms + t_crit * sem
    else:
        cum_lo = np.full_like(cum_ms, float("nan"))
        cum_hi = np.full_like(cum_ms, float("nan"))
    nodes = [
        OrtNodeTiming(
            index=i,
            name=ref.nodes[i].name,
            op_name=ref.nodes[i].op_name,
            dur_ms_mean=float(mean_ms[i]),
            dur_ms_std=float(across_std[i]),
            cum_ms_mean=float(cum_ms[i]),
            cum_ms_ci95_lo=float(cum_lo[i]),
            cum_ms_ci95_hi=float(cum_hi[i]),
        )
        for i in range(n_nodes)
    ]
    ort_mean = _mean_or_nan(trial_means)
    return OrtProfileResult(
        model=ref.model,
        providers=list(ref.providers),
        warmup_s=ref.warmup_s,
        runs_s=ref.runs_s,
        warmup_iters=ref.warmup_iters,
        runs_iters=ref.runs_iters,
        settle_iters=ref.settle_iters,
        n_nodes=n_nodes,
        profile_path=ref.profile_path,
        latency_ms=trial_means,
        latency_ms_mean=ort_mean,
        latency_ms_std=_std_or_zero(trial_means),
        latency_ms_p50=_percentile(trial_means, 50),
        latency_ms_p95=_percentile(trial_means, 95),
        nodes=nodes,
    )


def merge_ort_runtime_nvml_power(
    nvml_result: Any,
    ort: OrtProfileResult,
    *,
    nvml_results: list[Any] | None = None,
    ort_trials: list[OrtProfileResult] | None = None,
) -> dict[str, Any]:
    """Combine split-pass ORT runtime with NVML power from ``benchmark_onnx``.

    Primary ``latency_ms_*`` come from ORT kernel sums; ``power_w_*`` / clocks /
    temps from the separate NVML pass; ``energy_mj_*`` = ORT latency × NVML power.

    When ``ort_trials`` / ``nvml_results`` are provided (multi-trial batched
    schedule), aggregates across trials first.
    """
    if ort_trials is not None and len(ort_trials) > 0:
        ort = aggregate_ort_trials(ort_trials)
    if nvml_results is not None and len(nvml_results) > 0:
        power_trials = [float(r.power_w_mean) for r in nvml_results]
        nvml_result = nvml_results[-1]
        payload = nvml_result.to_dict()
        payload["power_w_mean"] = _mean_or_nan(power_trials)
        payload["power_w"] = power_trials
        payload["trials"] = len(nvml_results)
        # Average other steady scalars across NVML trials when present.
        for key in (
            "temperature_c_mean",
            "frequency_mhz_mean",
            "frequency_mem_mhz_mean",
            "frequency_graphics_mhz_mean",
            "frequency_video_mhz_mean",
        ):
            vals = [float(getattr(r, key)) for r in nvml_results]
            payload[key] = _mean_or_nan(vals)
    else:
        payload = nvml_result.to_dict()
        payload["trials"] = 1

    power_mean = float(payload["power_w_mean"])
    ort_mean = float(ort.latency_ms_mean)
    _attach_node_energy(ort.nodes, power_mean)

    payload["latency_source"] = "ort_profiler"
    payload["power_source"] = "nvml"
    payload["latency_ms"] = list(ort.latency_ms)
    payload["latency_ms_mean"] = ort_mean
    payload["latency_ms_std"] = float(ort.latency_ms_std)
    payload["latency_ms_p50"] = float(ort.latency_ms_p50)
    payload["latency_ms_p95"] = float(ort.latency_ms_p95)
    payload["throughput_inf_per_s"] = (1000.0 / ort_mean) if ort_mean > 0 else float("nan")

    if ort_mean == ort_mean and power_mean == power_mean and ort_mean > 0 and power_mean > 0:
        energy_mean = ort_mean * power_mean
    else:
        energy_mean = 0.0
    payload["energy_mj_mean"] = energy_mean
    payload["energy_mj"] = [lat * power_mean for lat in ort.latency_ms]
    payload["energy_mj_std"] = (
        statistics.pstdev(payload["energy_mj"]) if len(payload["energy_mj"]) > 1 else 0.0
    )
    payload["energy_mj_total"] = energy_mean * len(ort.latency_ms)
    payload["steady_skip_iters"] = 0
    payload["steady_iters"] = len(ort.latency_ms)
    payload["ort_profiler"] = ort.to_dict()
    if ort_trials is not None and len(ort_trials) > 1:
        payload["ort_profiler"]["trials"] = len(ort_trials)
    payload["backend"] = "onnxruntime+ort_profiler"
    return payload
