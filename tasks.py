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
def benchmark(ctx: Context, extra: str = "") -> None:
    """Run ONNX model benchmark locally (Hydra overrides via ``extra``).

    Example::

        invoke benchmark --extra="experiment.name=lop7 benchmark.device=4"
    """
    ctx.run(
        f"PYTHONPATH=src python src/{PROJECT_NAME}/benchmark.py {extra}",
        echo=True,
        pty=not WINDOWS,
    )


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


def _docker_run_benchmark(
    ctx: Context, script: str, device: int, extra: str, gpus: str = "all"
) -> None:
    """Run a benchmark script inside a bind-mounted container.

    Uses ``--gpus all`` (default) so CUDA/NVML indices match the host. Selection
    is done with ``benchmark.device``; do not use ``--gpus device=N`` if you need
    the host GPU number preserved (NVIDIA remaps a single exposed GPU to index 0).
    """
    env_file = " --env-file .env" if os.path.isfile(".env") else ""
    # CUDA_DEVICE_ORDER keeps enumeration aligned with nvidia-smi PCI order.
    ctx.run(
        f'docker run --rm --gpus "{gpus}" -v "$(pwd):/workspace" -w /workspace{env_file} '
        f"-e PYTHONPATH=/workspace/src -e CUDA_DEVICE_ORDER=PCI_BUS_ID "
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
            raise ValueError(
                f"Unsupported --gpus value {gpus!r}; use --device N with --gpus all"
            )
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
def docker_benchmark(
    ctx: Context, device: int = 4, extra: str = "", gpus: str = "all"
) -> None:
    """Benchmark in Docker with the repo bind-mounted (defaults to host GPU #4).

    All host GPUs stay visible with their original indices; ``--device`` maps to
    Hydra ``benchmark.device`` for inference and NVML energy metering.

    Example::

        invoke docker-benchmark --device=4 --extra="experiment.name=lop7"

    Legacy ``--gpus=device=4`` is accepted and converted to ``--gpus all`` + device 4.
    """
    device, gpus = _normalize_docker_gpu_selection(device, gpus)
    _docker_run_benchmark(
        ctx, f"src/{PROJECT_NAME}/benchmark.py", device=device, extra=extra, gpus=gpus
    )


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
