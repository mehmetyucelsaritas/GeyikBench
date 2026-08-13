import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "geyikbench"
PYTHON_VERSION = "3.12"
DOCKER_BENCHMARK_IMAGE = "geyikbench-benchmark:latest"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"PYTHONPATH=src python src/{PROJECT_NAME}/data.py",
        echo=True,
        pty=not WINDOWS,
    )


@task
def benchmark(ctx: Context, device: int = 0, extra: str = "", lock_clocks: bool = True) -> None:
    """Run ONNX model benchmark locally (Hydra overrides via ``extra``).

    Pins application clocks on ``--device`` only (default GPU 0). Example::

        invoke benchmark --device=0 --extra="experiment.name=lop7"
        invoke benchmark --no-lock-clocks --extra="experiment.name=lop7"
        invoke benchmark --extra="benchmark.use_ort_profiler=true experiment.name=lop7_ort"
        invoke benchmark --extra="benchmark.use_ort_profiler=true benchmark.trials=10"
    """
    import re

    # Prefer explicit --device; allow Hydra override in extra to win when present.
    match = re.search(r"benchmark\.device\s*=\s*(\d+)", extra)
    gpu = int(match.group(1)) if match else int(device)
    if lock_clocks:
        docker_lock_gpu_clocks(ctx, devices=str(gpu))
    # Keep Hydra device in sync when only --device was passed.
    if match is None and f"benchmark.device=" not in extra:
        extra = f"{extra} benchmark.device={gpu}".strip()
    ctx.run(
        f"PYTHONPATH=src python src/{PROJECT_NAME}/benchmark.py {extra}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def export_dataset(ctx: Context, runs: str, output: str) -> None:
    """Export profiling runs as a geyik training dataset (manifest.jsonl + ONNX copies).

    ``runs`` is a comma-separated list of run directories. Example::

        invoke export-dataset --runs=outputs/2026-07-30_19-32-44_lop7_1h --output=exports/lop7_full
    """
    run_args = " ".join(part.strip() for part in runs.split(",") if part.strip())
    ctx.run(
        f"PYTHONPATH=src python src/{PROJECT_NAME}/export_dataset.py --runs {run_args} --output {output}",
        echo=True,
        pty=not WINDOWS,
    )


# Readable local folder label for timing sweeps (date_timing_sweep_<wandb_id>).
WANDB_TIMING_SWEEP_NAME = "timing_sweep"


def _wandb_timing_sweep_dir(sweep_token: str | None = None) -> "Path":
    """Local root for W&B timing-sweep files (parent of the ``wandb/`` subdir).

    With a sweep id, returns ``outputs/<YYYY-MM-DD_HH-MM-SS>_timing_sweep_<id>/``
    (reuses an existing dated folder for that id when present).
    """
    import re
    from datetime import datetime
    from pathlib import Path

    root = Path("outputs")
    root.mkdir(parents=True, exist_ok=True)
    if not sweep_token:
        return root.resolve()

    token = sweep_token.rstrip("/").split("/")[-1]
    # Match date_timing_sweep_<id> (current) or date_<id> / bare legacy layouts.
    dated_re = re.compile(
        rf"^\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}"
        rf"(?:_{re.escape(WANDB_TIMING_SWEEP_NAME)})?_{re.escape(token)}$"
    )
    dated = sorted(
        (p for p in root.iterdir() if p.is_dir() and dated_re.match(p.name)),
        key=lambda p: p.name,
    )
    if dated:
        return dated[-1].resolve()
    legacy = root / "wandb_timing_sweep" / token
    if legacy.is_dir():
        return legacy.resolve()
    legacy_dated = (
        sorted((root / "wandb_timing_sweep").glob(f"*_{token}")) if (root / "wandb_timing_sweep").is_dir() else []
    )
    if legacy_dated:
        return legacy_dated[-1].resolve()
    bare = root / token
    if bare.is_dir():
        return bare.resolve()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = root / f"{stamp}_{WANDB_TIMING_SWEEP_NAME}_{token}"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _create_wandb_sweep(
    config: str = "configs/wandb/sweep_timing.yaml",
    project: str = "geyikbench",
    entity: str = "",
) -> str:
    """Create a W&B sweep and return ``entity/project/sweep_id`` (or ``project/id``)."""
    import re
    import shutil
    import subprocess
    from pathlib import Path

    env = os.environ.copy()
    if Path(".env").is_file():
        for line in Path(".env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'").strip('"')

    # Create under the shared parent; relocate into a dated sweep folder below.
    parent = _wandb_timing_sweep_dir()
    env["WANDB_DIR"] = str(parent)

    cmd = ["wandb", "sweep", "--project", project]
    if entity:
        cmd.extend(["--entity", entity])
    cmd.append(config)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(output, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wandb sweep failed with exit {proc.returncode}")

    # Prefer the full agent path from CLI output.
    match = re.search(r"wandb agent\s+(\S+/\S+/\S+)", output)
    if match:
        path = match.group(1).strip()
    else:
        match = re.search(r"Creating sweep with ID:\s*(\S+)", output)
        if not match:
            raise RuntimeError("Could not parse sweep id from wandb sweep output")
        sweep_id = match.group(1).strip()
        path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"

    token = path.rstrip("/").split("/")[-1]
    sweep_dir = _wandb_timing_sweep_dir(token)
    # Move ``wandb sweep`` metadata into the dated folder when it was written
    # under the shared parent (outputs/wandb/sweep-<id>).
    src_meta = parent / "wandb" / f"sweep-{token}"
    dst_meta = sweep_dir / "wandb" / f"sweep-{token}"
    if src_meta.is_dir() and not dst_meta.exists():
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_meta), str(dst_meta))
    print(f"Local sweep directory: {sweep_dir}", flush=True)
    return path


@task
def wandb_sweep_timing(
    ctx: Context,
    config: str = "configs/wandb/sweep_timing.yaml",
    project: str = "geyikbench",
    entity: str = "",
) -> None:
    """Create a W&B Sweep for warmup_s × runs_s repeatability.

    Prints the sweep id / URL. Then start workers with::

        invoke wandb-sweep-timing-agent --sweep-id=<id>

    Or create + run in one step::

        invoke wandb-sweep-timing-run
    """
    path = _create_wandb_sweep(config=config, project=project, entity=entity)
    print(f"Created sweep: {path}")
    print(f"Start agents with:\n  invoke wandb-sweep-timing-agent --sweep-id={path}")


@task
def wandb_sweep_timing_run(
    ctx: Context,
    config: str = "configs/wandb/sweep_timing.yaml",
    project: str = "geyikbench",
    entity: str = "",
    count: int = 0,
    devices: str = "all",
    lock_clocks: bool = True,
) -> None:
    """Create a fresh W&B timing sweep and start agents immediately."""
    path = _create_wandb_sweep(config=config, project=project, entity=entity)
    print(f"Created sweep: {path}")
    wandb_sweep_timing_agent(
        ctx,
        sweep_id=path,
        count=count,
        project=project,
        entity=entity,
        devices=devices,
        lock_clocks=lock_clocks,
    )


@task
def wandb_sweep_timing_agent(
    ctx: Context,
    sweep_id: str = "new",
    count: int = 0,
    project: str = "geyikbench",
    entity: str = "",
    devices: str = "all",
    config: str = "configs/wandb/sweep_timing.yaml",
    lock_clocks: bool = True,
) -> None:
    """Run W&B Sweep agent(s) for the timing sweep.

    ``--sweep-id=new`` (default) creates a fresh sweep first — use this when the
    previous sweep is finished. Pass an existing id only while that sweep is still
    running.

    ``devices=all`` launches one agent per visible GPU. By default pins equal
    application clocks on those same GPUs only (unused cards stay unlocked).

    Example::

        invoke wandb-sweep-timing-run
        invoke wandb-sweep-timing-agent --sweep-id=new
        invoke wandb-sweep-timing-agent --sweep-id=new --devices=0,1,2
        invoke wandb-sweep-timing-agent --sweep-id=entity/project/abc123
    """
    import subprocess
    from pathlib import Path

    if lock_clocks:
        docker_lock_gpu_clocks(ctx, devices=devices)

    if not sweep_id or sweep_id.strip().lower() in {"new", "create", "fresh"}:
        sweep_id = _create_wandb_sweep(config=config, project=project, entity=entity)
        print(f"Using newly created sweep: {sweep_id}")

    if devices.strip().lower() == "all":
        try:
            from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

            nvmlInit()
            try:
                gpu_ids = list(range(int(nvmlDeviceGetCount())))
            finally:
                nvmlShutdown()
        except Exception:
            gpu_ids = [0]
    else:
        gpu_ids = [int(x.strip()) for x in devices.split(",") if x.strip()]

    path = sweep_id if "/" in sweep_id else (f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}")
    sweep_token = path.rstrip("/").split("/")[-1]
    agent_cmd = ["wandb", "agent"]
    if count > 0:
        agent_cmd.extend(["--count", str(count)])
    agent_cmd.append(path)

    env_base = os.environ.copy()
    if Path(".env").is_file():
        # Lightweight .env load so WANDB_API_KEY is available to agents.
        for line in Path(".env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_base[key.strip()] = value.strip().strip("'").strip('"')
    env_base["PYTHONPATH"] = f"src{os.pathsep}{env_base['PYTHONPATH']}" if env_base.get("PYTHONPATH") else "src"
    env_base["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Agent + child runs write under outputs/<date>_<sweep>/wandb, not repo-root wandb/.
    env_base["WANDB_DIR"] = str(_wandb_timing_sweep_dir(sweep_token))

    print(f"Starting {len(gpu_ids)} wandb agent(s) on GPUs {gpu_ids} for {path}")
    print(f"Local W&B dir: {env_base['WANDB_DIR']}")
    procs: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for gpu in gpu_ids:
            log_path = Path(f"wandb_agent_gpu{gpu}.log")
            log_fh = log_path.open("w")
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(
                agent_cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            procs.append((gpu, proc, log_fh))
            print(f"  GPU {gpu}: pid={proc.pid} log={log_path}")

        failures = []
        for gpu, proc, log_fh in procs:
            code = proc.wait()
            log_fh.close()
            if code != 0:
                failures.append((gpu, code))
                print(f"[ERROR] agent on GPU {gpu} exited with {code} (see wandb_agent_gpu{gpu}.log)")
        if failures:
            hint = ""
            try:
                log0 = Path("wandb_agent_gpu0.log").read_text()
                if "is not running" in log0:
                    hint = (
                        "\nHint: this W&B sweep is finished/stopped. Re-run with a fresh sweep:\n"
                        "  invoke wandb-sweep-timing-run\n"
                        "or:\n"
                        "  invoke wandb-sweep-timing-agent --sweep-id=new"
                    )
            except OSError:
                pass
            raise RuntimeError(
                "wandb agent failed on GPU(s): " + ", ".join(f"{gpu} (exit {code})" for gpu, code in failures) + hint
            )
    except KeyboardInterrupt:
        print("Stopping agents...")
        for _, proc, log_fh in procs:
            proc.terminate()
            log_fh.close()
        raise


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


def _with_dotenv(command: str) -> str:
    """Prefix a shell command with ``.env`` loading when the file exists."""
    if os.path.isfile(".env"):
        return f"set -a && . ./.env && set +a && {command}"
    return command


def _docker_run_benchmark(ctx: Context, script: str, device: int, extra: str, gpus: str = "all") -> None:
    """Run a benchmark script inside a bind-mounted container.

    Uses ``--gpus all`` (default) so CUDA/NVML indices match the host. Selection
    is done with ``benchmark.device``; do not use ``--gpus device=N`` if you need
    the host GPU number preserved (NVIDIA remaps a single exposed GPU to index 0).
    """
    env_file = " --env-file .env" if os.path.isfile(".env") else ""
    # Match host uid/gid so bind-mounted outputs/ stay writable outside Docker.
    user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else ""
    user_flag = f" --user {user}" if user else ""
    # CUDA_DEVICE_ORDER keeps enumeration aligned with nvidia-smi PCI order.
    # HOME=/tmp avoids permission errors when the image user has no writable $HOME.
    ctx.run(
        f'docker run --rm --gpus "{gpus}"{user_flag} -v "$(pwd):/workspace" -w /workspace{env_file} '
        f"-e PYTHONPATH=/workspace/src -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOME=/tmp "
        f"{DOCKER_BENCHMARK_IMAGE} {script} benchmark.device={device} {extra}",
        echo=True,
        pty=not WINDOWS,
    )


def _normalize_docker_gpu_selection(device: int, gpus: str) -> tuple[int, str]:
    """Translate legacy ``--gpus device=N`` into ``--gpus all`` + ``--device N``.

    NVIDIA Container Toolkit remaps a single exposed GPU to container index 0.
    To keep host GPU numbers, always expose all GPUs and select with ``--device``.
    """
    value = gpus.strip()
    if value.startswith("device="):
        selected = value.split("=", 1)[1].split(",")[0].strip()
        if not selected.isdigit():
            raise ValueError(f"Unsupported --gpus value {gpus!r}; use --device N with --gpus all")
        return int(selected), "all"
    return device, value


@task
def docker_build_benchmark(ctx: Context, progress: str = "plain") -> None:
    """Build the GPU benchmark image (rebuild only when deps change)."""
    ctx.run(
        f"docker build -t {DOCKER_BENCHMARK_IMAGE} . -f dockerfiles/benchmark.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_lock_gpu_clocks(
    ctx: Context,
    mem: int = 5705,
    graphics: int = 1404,
    power: int = 0,
    reset: bool = False,
    devices: str = "all",
    image: str = "nvidia/cuda:12.4.1-base-ubuntu22.04",
) -> None:
    """Pin GPU application clocks via privileged Docker (no host sudo).

    ``devices`` is ``all`` or a comma-separated host GPU list (e.g. ``0`` / ``0,1``).
    Application clocks apply only under CUDA load — idle GPUs still downclock.

    Example::

        invoke docker-lock-gpu-clocks --devices=0
        invoke docker-lock-gpu-clocks --devices=all
        invoke docker-lock-gpu-clocks --reset --devices=0
    """
    env = f"-e MEM_MHZ={mem} -e GRAPHICS_MHZ={graphics} -e RESET={'1' if reset else '0'} -e GPU_IDS={devices}"
    if power > 0:
        env += f" -e POWER_W={power}"
    # Root inside privileged container can call nvidia-smi clock/power APIs.
    ctx.run(
        f"docker run --rm --privileged --gpus all {env} "
        f'-v "$(pwd)/scripts/lock_gpu_clocks.sh:/lock_gpu_clocks.sh:ro" '
        f"--entrypoint bash {image} /lock_gpu_clocks.sh",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_benchmark(
    ctx: Context,
    device: int = 0,
    extra: str = "",
    gpus: str = "all",
    lock_clocks: bool = True,
) -> None:
    """Benchmark in Docker with the repo bind-mounted (defaults to host GPU #0).

    Pins application clocks on ``--device`` only by default. Legacy
    ``--gpus=device=0`` is converted to ``--gpus all`` + device selection.

    Example::

        invoke docker-benchmark --device=0 --extra="experiment.name=lop7"
    """
    device, gpus = _normalize_docker_gpu_selection(device, gpus)
    if lock_clocks:
        docker_lock_gpu_clocks(ctx, devices=str(device))
    _docker_run_benchmark(ctx, f"src/{PROJECT_NAME}/benchmark.py", device=device, extra=extra, gpus=gpus)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images."""
    docker_build_benchmark(ctx, progress=progress)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
