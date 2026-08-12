"""Export profiling results as a dataset for the geyik runtime estimator.

Walks benchmark run directories, joins each ``result.json`` with the profiled
ONNX model, and writes a self-contained export directory::

    exports/<name>/
    ├── manifest.jsonl   # one record per profiled model (runtime + energy labels)
    └── models/          # copies of the profiled .onnx files

Models are identified by the SHA-256 of their ONNX bytes so exports remain
valid across machines. The geyik repo consumes this layout with its
``ingest-geyikbench`` command.

Example::

    python src/geyikbench/export_dataset.py \
        --runs outputs/2026-07-30_19-32-44_lop7_1h --output exports/lop7_full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_METRIC_KEYS = [
    "latency_ms_mean",
    "latency_ms_std",
    "latency_ms_p50",
    "latency_ms_p95",
    "energy_mj_mean",
    "energy_mj_std",
    "power_w_mean",
    "throughput_inf_per_s",
    "warmup_s",
    "runs_s",
    "runs_iters",
    "steady_iters",
    "device_index",
    "device_name",
    "backend",
]


def find_result_files(run_dirs: list[Path]) -> list[Path]:
    """Collect result.json files from run directories (recursively), skipping wandb dirs."""
    results: list[Path] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        direct = run_dir / "result.json"
        if direct.is_file():
            results.append(direct)
            continue
        for path in sorted(run_dir.rglob("result.json")):
            if "wandb" in path.parts or "trials" in path.parts:
                continue
            results.append(path)
    return results


def resolve_model_path(recorded: str) -> Path | None:
    """Resolve the ONNX path recorded in a result payload.

    Tries the recorded path as-is and relative to the project root, then falls
    back to matching the basename anywhere under ``models/`` and ``data/``
    (profiling often ran on a different machine or the files moved since).
    """
    candidate = Path(recorded)
    if candidate.is_file():
        return candidate
    relative = PROJECT_ROOT / recorded
    if relative.is_file():
        return relative
    basename = candidate.name
    for search_root in (PROJECT_ROOT / "models", PROJECT_ROOT / "data"):
        if not search_root.is_dir():
            continue
        matches = sorted(search_root.rglob(basename))
        if matches:
            return matches[0]
    return None


def onnx_batch_size(onnx_path: Path) -> int:
    """Read the batch dimension of the first graph input from an ONNX file."""
    import onnx

    model = onnx.load(str(onnx_path))
    for graph_input in model.graph.input:
        dims = graph_input.type.tensor_type.shape.dim
        if dims and dims[0].HasField("dim_value"):
            return dims[0].dim_value
    return 1


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(payload: dict, result_path: Path, model_file: str, model_sha256: str, batch_size: int) -> dict:
    """Build one manifest record from a result payload."""
    experiment = payload.get("experiment") or {}
    run_dir = result_path.parent.resolve()
    source_run = str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir)
    record = {
        "model_file": model_file,
        "model_sha256": model_sha256,
        "batch_size": batch_size,
        "experiment_name": experiment.get("name", ""),
        "source_run": source_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in MANIFEST_METRIC_KEYS:
        record[key] = payload.get(key)
    return record


def export_dataset(run_dirs: list[Path], output_dir: Path) -> list[dict]:
    """Export profiling runs into a manifest + models directory for geyik.

    Args:
        run_dirs: Benchmark run directories to scan for result.json files.
        output_dir: Export directory to create (manifest.jsonl + models/).

    Returns:
        The list of manifest records written.
    """
    result_files = find_result_files(run_dirs)
    if not result_files:
        raise FileNotFoundError(f"No result.json files found under: {[str(d) for d in run_dirs]}")

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_hashes: set[str] = set()
    for result_path in result_files:
        payload = json.loads(result_path.read_text())
        recorded_model = payload.get("model", "")
        model_path = resolve_model_path(recorded_model)
        if model_path is None:
            print(f"[skip] cannot resolve model '{recorded_model}' for {result_path}", file=sys.stderr)
            continue

        model_sha256 = sha256_of(model_path)
        if model_sha256 in seen_hashes:
            print(f"[skip] duplicate model {model_path.name} ({model_sha256[:8]}) from {result_path}")
            continue
        seen_hashes.add(model_sha256)

        target_name = f"{model_path.stem}.onnx"
        target_path = models_dir / target_name
        if target_path.exists() and sha256_of(target_path) != model_sha256:
            target_name = f"{model_path.stem}_{model_sha256[:8]}.onnx"
            target_path = models_dir / target_name
        if not target_path.exists():
            shutil.copy2(model_path, target_path)

        record = build_record(
            payload,
            result_path,
            model_file=f"models/{target_name}",
            model_sha256=model_sha256,
            batch_size=onnx_batch_size(model_path),
        )
        records.append(record)
        print(f"[ok] {model_path.name} ({model_sha256[:8]}) latency={record['latency_ms_mean']:.3f} ms")

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {manifest_path}")
    return records


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="Benchmark run directories (searched recursively for result.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Export directory to create (manifest.jsonl + models/)",
    )
    args = parser.parse_args()
    export_dataset(args.runs, args.output)


if __name__ == "__main__":
    main()
