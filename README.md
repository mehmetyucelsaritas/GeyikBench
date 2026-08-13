# GeyikBench

GPU ONNX benchmarking (NVML power + latency) for geyik training labels.

## Benchmark

```bash
# Wall-clock latency + NVML power (default)
invoke benchmark --device=0 --extra="experiment.name=lop7"

# Split passes: ORT runtime, then NVML power (same warmup_s / runs_s each)
invoke benchmark --device=0 --extra="benchmark.use_ort_profiler=true experiment.name=lop7_ort"

# Multi-trial (all ORT first, then all NVML power) — set in YAML or override:
invoke benchmark --device=0 --extra="benchmark.use_ort_profiler=true benchmark.trials=10"
```

Config: ``configs/benchmark/benchmark.yaml`` (`trials`, `use_ort_profiler`, …).

With ``use_ort_profiler=true`` and e.g. ``warmup_s=5 runs_s=5``:

1. **~10s × trials** ORT profiler → ``latency_ms_*`` + per-node ``ort_profiler.nodes``
2. **~10s × trials** ``benchmark_onnx`` (no profiler) → ``power_w_*``

``energy_mj_* = ort_latency_ms * power_w``.
