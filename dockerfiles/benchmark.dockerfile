# GPU benchmark image: install Python deps only; mount source at runtime.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN grep -v '^torch' requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir --break-system-packages -r requirements-docker.txt

ENV PYTHONPATH=/workspace/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-u"]
CMD ["src/geyikbench/benchmark.py"]
