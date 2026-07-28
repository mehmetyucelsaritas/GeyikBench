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

import numpy as np
import typer
from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTotalEnergyConsumption,
    nvmlInit,
    nvmlShutdown,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


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
    warmup: int
    runs: int
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    energy_mj_mean: float
    energy_mj_std: float
    energy_mj_total: float
    power_w_mean: float
    throughput_inf_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NvmlEnergyMeter:
    """Measure GPU energy with NVML total-energy counters (mJ)."""

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
        typer.echo(
            f"[WARN] CUDAExecutionProvider unavailable ({available}); using CPU.",
            err=True,
        )
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    active = session.get_providers()
    typer.echo(f"[INFO] ORT providers: {active}", err=True)
    if "CUDAExecutionProvider" not in active:
        typer.echo(
            "[WARN] Running on CPU — GPU energy numbers will mostly reflect idle draw.",
            err=True,
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


def benchmark_onnx(
    model_path: Path,
    device_index: int = 0,
    warmup: int = 10,
    runs: int = 50,
    batch_size: int = 1,
) -> BenchmarkResult:
    meter = NvmlEnergyMeter(device_index)
    session = _create_ort_session(model_path, device_index)
    feeds = _make_onnx_feeds(session, batch_size)
    output_names = [out.name for out in session.get_outputs()]
    sync = lambda: _cuda_synchronize(device_index)

    # Warmup so kernels / clocks settle before measurement.
    for _ in range(warmup):
        session.run(output_names, feeds)
    sync()

    latencies_ms: list[float] = []
    energies_mj: list[float] = []
    powers_w: list[float] = []

    for _ in range(runs):
        sync()
        e0 = meter.energy_mj()
        p0 = meter.power_w()
        t0 = time.perf_counter()
        session.run(output_names, feeds)
        sync()
        t1 = time.perf_counter()
        e1 = meter.energy_mj()
        p1 = meter.power_w()

        latencies_ms.append((t1 - t0) * 1000.0)
        energies_mj.append(float(e1 - e0))
        powers_w.append(0.5 * (p0 + p1))

    device_name = meter.device_name
    meter.close()
    return _summarize(
        model=str(model_path),
        device_index=device_index,
        device_name=device_name,
        backend="onnxruntime",
        providers=list(session.get_providers()),
        warmup=warmup,
        runs=runs,
        latencies_ms=latencies_ms,
        energies_mj=energies_mj,
        powers_w=powers_w,
    )


def benchmark_callable(
    run_once: Any,
    *,
    model_label: str,
    device_index: int = 0,
    warmup: int = 10,
    runs: int = 50,
    backend: str = "callable",
    synchronize: Any | None = None,
) -> BenchmarkResult:
    """Benchmark an arbitrary inference callable with NVML energy + wall time."""
    meter = NvmlEnergyMeter(device_index)

    for _ in range(warmup):
        run_once()
        if synchronize is not None:
            synchronize()

    latencies_ms: list[float] = []
    energies_mj: list[float] = []
    powers_w: list[float] = []

    for _ in range(runs):
        if synchronize is not None:
            synchronize()
        e0 = meter.energy_mj()
        p0 = meter.power_w()
        t0 = time.perf_counter()
        run_once()
        if synchronize is not None:
            synchronize()
        t1 = time.perf_counter()
        e1 = meter.energy_mj()
        p1 = meter.power_w()

        latencies_ms.append((t1 - t0) * 1000.0)
        energies_mj.append(float(e1 - e0))
        powers_w.append(0.5 * (p0 + p1))

    device_name = meter.device_name
    meter.close()
    return _summarize(
        model=model_label,
        device_index=device_index,
        device_name=device_name,
        backend=backend,
        providers=[],
        warmup=warmup,
        runs=runs,
        latencies_ms=latencies_ms,
        energies_mj=energies_mj,
        powers_w=powers_w,
    )


def _summarize(
    *,
    model: str,
    device_index: int,
    device_name: str,
    backend: str,
    providers: list[str],
    warmup: int,
    runs: int,
    latencies_ms: list[float],
    energies_mj: list[float],
    powers_w: list[float],
) -> BenchmarkResult:
    mean_latency = statistics.fmean(latencies_ms)
    return BenchmarkResult(
        model=model,
        device_index=device_index,
        device_name=device_name,
        backend=backend,
        providers=providers,
        warmup=warmup,
        runs=runs,
        latency_ms_mean=mean_latency,
        latency_ms_std=statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        latency_ms_p50=_percentile(latencies_ms, 50),
        latency_ms_p95=_percentile(latencies_ms, 95),
        energy_mj_mean=statistics.fmean(energies_mj),
        energy_mj_std=statistics.pstdev(energies_mj) if len(energies_mj) > 1 else 0.0,
        energy_mj_total=float(sum(energies_mj)),
        power_w_mean=statistics.fmean(powers_w),
        throughput_inf_per_s=(1000.0 / mean_latency) if mean_latency > 0 else float("nan"),
    )


@app.command()
def main(
    model: Path = typer.Argument(..., exists=True, readable=True, help="Path to ONNX model"),
    device: int = typer.Option(0, "--device", "-d", help="CUDA / NVML device index"),
    warmup: int = typer.Option(10, "--warmup", "-w", help="Warmup iterations"),
    runs: int = typer.Option(50, "--runs", "-n", help="Measured iterations"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size for synthetic inputs"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional JSON report path"),
) -> None:
    """Measure inference latency and GPU energy for an ONNX model using NVML."""
    if model.suffix.lower() != ".onnx":
        raise typer.BadParameter("Only ONNX models are supported by the CLI for now.")

    result = benchmark_onnx(
        model_path=model,
        device_index=device,
        warmup=warmup,
        runs=runs,
        batch_size=batch_size,
    )
    payload = result.to_dict()
    typer.echo(json.dumps(payload, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"Wrote {output}", err=True)


if __name__ == "__main__":
    app()
