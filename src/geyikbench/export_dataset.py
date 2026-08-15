"""Export profiling results as a dataset for the geyik runtime estimator.

Walks benchmark run directories, joins each ``result.json`` with the profiled
ONNX model, and writes a self-contained export directory::

    exports/<name>/
    ├── manifest.jsonl   # one record per profiled model (runtime + energy labels)
    ├── models/          # copies of the profiled .onnx files
    └── nodes/           # per-node cumulative runtime/energy (ORT profiler runs)

Runs profiled with the ORT profiler also carry per-node cumulative labels. Those
are written to ``nodes/<stem>.json`` in ORT execution order and referenced from
the manifest via ``nodes_file``, so geyik can expand one model into one training
sample per node prefix.

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

from geyikbench.ort_profiler import normalize_onnx_node_name

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
    "latency_source",
    "power_source",
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


NODE_LABEL_KEYS = [
    "index",
    "onnx_node",
    "name",
    "op_name",
    "dur_ms_mean",
    "dur_ms_std",
    "cum_ms_mean",
    "cum_ms_ci95_lo",
    "cum_ms_ci95_hi",
    "energy_mj_mean",
    "energy_mj_std",
    "cum_mj_mean",
    "cum_mj_ci95_lo",
    "cum_mj_ci95_hi",
]


def build_record(
    payload: dict,
    result_path: Path,
    model_file: str,
    model_sha256: str,
    batch_size: int,
    nodes_file: str | None = None,
    n_nodes: int | None = None,
) -> dict:
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
    if nodes_file is not None:
        record["nodes_file"] = nodes_file
        record["n_nodes"] = n_nodes
    return record


def extract_node_labels(payload: dict) -> list[dict]:
    """Return per-node cumulative labels in ORT execution order, or [] when absent.

    Node records keep only the fields geyik needs to build prefix samples, with
    ``onnx_node`` backfilled from the raw kernel name for older results.
    """
    nodes = (payload.get("ort_profiler") or {}).get("nodes") or []
    labels: list[dict] = []
    for node in nodes:
        record = {key: node[key] for key in NODE_LABEL_KEYS if key in node}
        if not record.get("onnx_node"):
            record["onnx_node"] = normalize_onnx_node_name(node.get("name", ""))
        labels.append(record)
    return labels


def write_node_labels(nodes: list[dict], nodes_dir: Path, stem: str) -> str:
    """Write per-node cumulative labels next to the models and return the relative path."""
    nodes_dir.mkdir(parents=True, exist_ok=True)
    target = nodes_dir / f"{stem}.json"
    target.write_text(json.dumps(nodes, indent=2) + "\n", encoding="utf-8")
    return f"nodes/{target.name}"


def export_dataset(run_dirs: list[Path], output_dir: Path) -> list[dict]:
    """Export profiling runs into a manifest + models directory for geyik.

    Args:
        run_dirs: Benchmark run directories to scan for result.json files.
        output_dir: Export directory to create (manifest.jsonl + models/ + nodes/).

    Returns:
        The list of manifest records written.
    """
    result_files = find_result_files(run_dirs)
    if not result_files:
        raise FileNotFoundError(f"No result.json files found under: {[str(d) for d in run_dirs]}")

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir = output_dir / "nodes"

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

        node_labels = extract_node_labels(payload)
        nodes_file = None
        if node_labels:
            nodes_file = write_node_labels(node_labels, nodes_dir, Path(target_name).stem)

        record = build_record(
            payload,
            result_path,
            model_file=f"models/{target_name}",
            model_sha256=model_sha256,
            batch_size=onnx_batch_size(model_path),
            nodes_file=nodes_file,
            n_nodes=len(node_labels) or None,
        )
        records.append(record)
        node_note = f" nodes={len(node_labels)}" if node_labels else ""
        print(f"[ok] {model_path.name} ({model_sha256[:8]}) latency={record['latency_ms_mean']:.3f} ms{node_note}")

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
