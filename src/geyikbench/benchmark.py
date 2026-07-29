"""Benchmark ONNX / PyTorch model inference time and GPU energy via NVIDIA NVML."""

from __future__ import annotations

import ctypes
import glob
import json
import os
import site
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pynvml import (
    NVML_CLOCK_SM,
    NVML_TEMPERATURE_GPU,
    NVMLError,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetTotalEnergyConsumption,
    nvmlInit,
    nvmlShutdown,
)


def _nvidia_lib_dirs() -> list[str]:
    search_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(user_site)
    lib_dirs: list[str] = []
    for root in search_roots:
        lib_dirs.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))
    return lib_dirs


def _ensure_nvidia_library_path() -> None:
    """Re-exec with LD_LIBRARY_PATH so ORT can resolve pip CUDA/cuDNN libs."""
    if os.environ.get("GEYIKBENCH_NVLIB_READY") == "1":
        return
    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, current] if current else lib_dirs)
    os.environ["GEYIKBENCH_NVLIB_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _preload_nvidia_libs() -> list[str]:
    """Load pip-provided NVIDIA CUDA libs so ORT's CUDA EP can resolve them."""
    loaded: list[str] = []
    for lib_dir in _nvidia_lib_dirs():
        for path in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
            # Skip broken symlinks / python packaging metadata.
            if not os.path.isfile(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(path)
            except OSError:
                continue
    return loaded


_ensure_nvidia_library_path()
_preload_nvidia_libs()


@dataclass
class BenchmarkResult:
    model: str
    device_index: int
    device_name: str
    backend: str
    providers: list[str]
    warmup_s: float
    runs_s: float
    warmup_iters: int
    runs_iters: int
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    energy_mj_mean: float
    energy_mj_std: float
    energy_mj_total: float
    power_w_mean: float
    temperature_c_mean: float
    frequency_mhz_mean: float
    throughput_inf_per_s: float
    # One sample per measured inference run (same length / order).
    latency_ms: list[float]
    energy_mj: list[float]
    power_w: list[float]
    temperature_c: list[float]
    frequency_mhz: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NvmlEnergyMeter:
    """Measure GPU energy, power, temperature, and SM clocks via NVML."""

    def __init__(self, device_index: int = 0) -> None:
        nvmlInit()
        self.device_index = device_index
        self.handle = nvmlDeviceGetHandleByIndex(device_index)
        self.device_name = nvmlDeviceGetName(self.handle)
        if isinstance(self.device_name, bytes):
            self.device_name = self.device_name.decode("utf-8")

    def energy_mj(self) -> int:
        return int(nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def power_w(self) -> float:
        return nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def temperature_c(self) -> float:
        return float(nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU))

    def frequency_mhz(self) -> float:
        return float(nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM))

    def close(self) -> None:
        try:
            nvmlShutdown()
        except NVMLError:
            pass


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _make_onnx_feeds(
    session: Any, batch_size: int, override_shapes: dict[str, list[int]] | None = None
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = list(inp.shape)
        for i, dim in enumerate(shape):
            if isinstance(dim, str) or dim is None:
                shape[i] = batch_size if i == 0 else 144
        if override_shapes and inp.name in override_shapes:
            shape = list(override_shapes[inp.name])
        dtype = np.float32
        if inp.type == "tensor(float16)":
            dtype = np.float16
        elif inp.type == "tensor(int64)":
            dtype = np.int64
        elif inp.type == "tensor(int32)":
            dtype = np.int32
        feeds[inp.name] = np.random.randn(*shape).astype(dtype)
    return feeds


def _create_ort_session(model_path: Path, device_index: int) -> Any:
    import onnxruntime as ort

    providers: list[tuple[str, dict[str, Any]] | str] = [
        (
            "CUDAExecutionProvider",
            {"device_id": device_index},
        ),
        "CPUExecutionProvider",
    ]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        print(
            f"[WARN] CUDAExecutionProvider unavailable ({available}); using CPU.",
            file=sys.stderr,
        )
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    active = session.get_providers()
    print(f"[INFO] ORT providers: {active}", file=sys.stderr)
    if "CUDAExecutionProvider" not in active:
        print(
            "[WARN] Running on CPU — GPU energy numbers will mostly reflect idle draw.",
            file=sys.stderr,
        )
    return session


def _cuda_synchronize(device_index: int) -> None:
    """Best-effort GPU sync so timers cover completed device work."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device_index)
    except Exception:
        pass


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _run_until(
    step: Any,
    *,
    duration_s: float | None = None,
    max_iters: int | None = None,
) -> int:
    """Call ``step`` until a time and/or iteration limit is hit. Returns iterations.

    Stops at whichever configured limit is reached first. At least one of
    ``duration_s`` / ``max_iters`` must be set.
    """
    if duration_s is None and max_iters is None:
        raise ValueError("Provide duration_s and/or max_iters")
    if duration_s is not None and duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s}")
    if max_iters is not None and max_iters < 1:
        raise ValueError(f"max_iters must be >= 1 when set, got {max_iters}")

    n = 0
    deadline = time.perf_counter() + duration_s if duration_s is not None else None
    while True:
        step()
        n += 1
        if max_iters is not None and n >= max_iters:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
    return n


def benchmark_onnx(
    model_path: Path,
    device_index: int = 0,
    warmup_s: float | None = 15.0,
    runs_s: float | None = 30.0,
    warmup_iters: int | None = None,
    runs_iters: int | None = None,
    batch_size: int = 1,
) -> BenchmarkResult:
    meter = NvmlEnergyMeter(device_index)
    session = _create_ort_session(model_path, device_index)
    feeds = _make_onnx_feeds(session, batch_size)
    output_names = [out.name for out in session.get_outputs()]
    sync = lambda: _cuda_synchronize(device_index)

    # Warmup so kernels / clocks settle before measurement.
    warmup_iters_done = _run_until(
        lambda: session.run(output_names, feeds),
        duration_s=warmup_s,
        max_iters=warmup_iters,
    )
    sync()

    latencies_ms: list[float] = []
    energies_mj: list[float] = []
    powers_w: list[float] = []
    temperatures_c: list[float] = []
    frequencies_mhz: list[float] = []

    def _measured_once() -> None:
        sync()
        e0 = meter.energy_mj()
        p0 = meter.power_w()
        temp0 = meter.temperature_c()
        freq0 = meter.frequency_mhz()
        t0 = time.perf_counter()
        session.run(output_names, feeds)
        sync()
        t1 = time.perf_counter()
        e1 = meter.energy_mj()
        p1 = meter.power_w()
        temp1 = meter.temperature_c()
        freq1 = meter.frequency_mhz()

        latencies_ms.append((t1 - t0) * 1000.0)
        energies_mj.append(float(e1 - e0))
        powers_w.append(0.5 * (p0 + p1))
        temperatures_c.append(0.5 * (temp0 + temp1))
        frequencies_mhz.append(0.5 * (freq0 + freq1))

    runs_iters_done = _run_until(
        _measured_once,
        duration_s=runs_s,
        max_iters=runs_iters,
    )

    device_name = meter.device_name
    meter.close()
    return _summarize(
        model=str(model_path),
        device_index=device_index,
        device_name=device_name,
        backend="onnxruntime",
        providers=list(session.get_providers()),
        warmup_s=warmup_s,
        runs_s=runs_s,
        warmup_iters=warmup_iters_done,
        runs_iters=runs_iters_done,
        latencies_ms=latencies_ms,
        energies_mj=energies_mj,
        powers_w=powers_w,
        temperatures_c=temperatures_c,
        frequencies_mhz=frequencies_mhz,
    )


def benchmark_callable(
    run_once: Any,
    *,
    model_label: str,
    device_index: int = 0,
    warmup_s: float | None = 15.0,
    runs_s: float | None = 30.0,
    warmup_iters: int | None = None,
    runs_iters: int | None = None,
    backend: str = "callable",
    synchronize: Any | None = None,
) -> BenchmarkResult:
    """Benchmark an arbitrary inference callable with NVML energy + wall time."""
    meter = NvmlEnergyMeter(device_index)

    def _warmup_once() -> None:
        run_once()
        if synchronize is not None:
            synchronize()

    warmup_iters_done = _run_until(
        _warmup_once,
        duration_s=warmup_s,
        max_iters=warmup_iters,
    )

    latencies_ms: list[float] = []
    energies_mj: list[float] = []
    powers_w: list[float] = []
    temperatures_c: list[float] = []
    frequencies_mhz: list[float] = []

    def _measured_once() -> None:
        if synchronize is not None:
            synchronize()
        e0 = meter.energy_mj()
        p0 = meter.power_w()
        temp0 = meter.temperature_c()
        freq0 = meter.frequency_mhz()
        t0 = time.perf_counter()
        run_once()
        if synchronize is not None:
            synchronize()
        t1 = time.perf_counter()
        e1 = meter.energy_mj()
        p1 = meter.power_w()
        temp1 = meter.temperature_c()
        freq1 = meter.frequency_mhz()

        latencies_ms.append((t1 - t0) * 1000.0)
        energies_mj.append(float(e1 - e0))
        powers_w.append(0.5 * (p0 + p1))
        temperatures_c.append(0.5 * (temp0 + temp1))
        frequencies_mhz.append(0.5 * (freq0 + freq1))

    runs_iters_done = _run_until(
        _measured_once,
        duration_s=runs_s,
        max_iters=runs_iters,
    )

    device_name = meter.device_name
    meter.close()
    return _summarize(
        model=model_label,
        device_index=device_index,
        device_name=device_name,
        backend=backend,
        providers=[],
        warmup_s=warmup_s,
        runs_s=runs_s,
        warmup_iters=warmup_iters_done,
        runs_iters=runs_iters_done,
        latencies_ms=latencies_ms,
        energies_mj=energies_mj,
        powers_w=powers_w,
        temperatures_c=temperatures_c,
        frequencies_mhz=frequencies_mhz,
    )


def _summarize(
    *,
    model: str,
    device_index: int,
    device_name: str,
    backend: str,
    providers: list[str],
    warmup_s: float | None,
    runs_s: float | None,
    warmup_iters: int,
    runs_iters: int,
    latencies_ms: list[float],
    energies_mj: list[float],
    powers_w: list[float],
    temperatures_c: list[float],
    frequencies_mhz: list[float],
) -> BenchmarkResult:
    mean_latency = statistics.fmean(latencies_ms)
    return BenchmarkResult(
        model=model,
        device_index=device_index,
        device_name=device_name,
        backend=backend,
        providers=providers,
        warmup_s=warmup_s if warmup_s is not None else float("nan"),
        runs_s=runs_s if runs_s is not None else float("nan"),
        warmup_iters=warmup_iters,
        runs_iters=runs_iters,
        latency_ms_mean=mean_latency,
        latency_ms_std=statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        latency_ms_p50=_percentile(latencies_ms, 50),
        latency_ms_p95=_percentile(latencies_ms, 95),
        energy_mj_mean=statistics.fmean(e for e in energies_mj if e > 0) if any(e > 0 for e in energies_mj) else 0.0,
        energy_mj_std=statistics.pstdev(e for e in energies_mj if e > 0) if sum(1 for e in energies_mj if e > 0) > 1 else 0.0,
        energy_mj_total=float(sum(energies_mj)),
        power_w_mean=statistics.fmean(powers_w),
        temperature_c_mean=statistics.fmean(temperatures_c),
        frequency_mhz_mean=statistics.fmean(frequencies_mhz),
        throughput_inf_per_s=(1000.0 / mean_latency) if mean_latency > 0 else float("nan"),
        latency_ms=latencies_ms,
        energy_mj=energies_mj,
        power_w=powers_w,
        temperature_c=temperatures_c,
        frequency_mhz=frequencies_mhz,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Measure inference latency and GPU energy for an ONNX model using NVML."""
    bench = cfg.benchmark
    model = Path(to_absolute_path(str(bench.model)))
    if model.suffix.lower() != ".onnx":
        raise ValueError(f"Only ONNX models are supported for now, got: {model}")
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")

    np.random.seed(int(cfg.experiment.seed))

    result = benchmark_onnx(
        model_path=model,
        device_index=int(bench.device),
        warmup_s=_optional_float(OmegaConf.select(bench, "warmup_s")),
        runs_s=_optional_float(OmegaConf.select(bench, "runs_s")),
        warmup_iters=_optional_int(OmegaConf.select(bench, "warmup_iters")),
        runs_iters=_optional_int(OmegaConf.select(bench, "runs_iters")),
        # Optional; defaults to single-sample when omitted.
        batch_size=int(OmegaConf.select(bench, "batch_size", default=1)),
    )
    payload = {
        "experiment": OmegaConf.to_container(cfg.experiment, resolve=True),
        **result.to_dict(),
    }
    print(json.dumps(payload, indent=2))

    if bench.output is not None:
        output = Path(to_absolute_path(str(bench.output)))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {output}", file=sys.stderr)

        # Log to W&B from within this process so the wandb/ folder is written
        # into the same Hydra output dir (which is writable by the current process).
        if OmegaConf.select(cfg, "wandb.enabled", default=True):
            try:
                from geyikbench.visualize import log_result
                log_result(
                    result_path=output,
                    project=OmegaConf.select(cfg, "wandb.project", default="geyikbench"),
                    wandb_dir=output.parent,
                )
            except Exception as exc:  # never crash the benchmark over W&B
                print(f"[WARN] W&B logging failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
