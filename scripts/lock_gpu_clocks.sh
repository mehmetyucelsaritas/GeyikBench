#!/usr/bin/env bash
# Pin identical *application* clocks on selected GPUs (root / privileged).
#
# Application clocks only apply while a CUDA workload is running. Idle GPUs
# still drop to P8 (~139 MHz) — they do not stay at the pinned graphics clock.
# TITAN Xp / Pascal do not support nvidia-smi --lock-gpu-clocks; -ac is the
# supported way to keep under-load graphics clocks aligned across cards.
#
# GPU_IDS=all (default) or a comma-separated list, e.g. GPU_IDS=0 or GPU_IDS=0,1
set -euo pipefail

MEM_MHZ="${MEM_MHZ:-5705}"
GRAPHICS_MHZ="${GRAPHICS_MHZ:-1404}"
POWER_W="${POWER_W:-}" # optional, e.g. 250
RESET="${RESET:-0}"
GPU_IDS="${GPU_IDS:-all}"

echo "GPUs:"
nvidia-smi -L

gpu_args=()
if [[ "${GPU_IDS}" != "all" ]]; then
  # nvidia-smi -i accepts comma-separated indices.
  gpu_args=(-i "${GPU_IDS}")
  echo "Target GPU(s): ${GPU_IDS}"
else
  echo "Target GPU(s): all"
fi

if [[ "${RESET}" == "1" ]]; then
  echo "Resetting application clocks and power limits to defaults..."
  nvidia-smi "${gpu_args[@]}" -rac
  nvidia-smi "${gpu_args[@]}" -rpl || true
else
  # Persistence keeps the driver loaded so -ac settings survive between runs
  # without forcing high clocks while idle.
  echo "Enabling persistence mode..."
  nvidia-smi "${gpu_args[@]}" -pm 1

  echo "Setting application clocks MEM=${MEM_MHZ} GRAPHICS=${GRAPHICS_MHZ}..."
  nvidia-smi "${gpu_args[@]}" -ac "${MEM_MHZ},${GRAPHICS_MHZ}"

  if [[ -n "${POWER_W}" ]]; then
    echo "Setting power limit to ${POWER_W} W..."
    nvidia-smi "${gpu_args[@]}" -pl "${POWER_W}"
  fi
fi

echo
echo "Configured application clocks / power:"
nvidia-smi "${gpu_args[@]}" --query-gpu=index,name,clocks.applications.graphics,clocks.applications.memory,power.limit --format=csv
echo "Current (idle) clocks:"
nvidia-smi "${gpu_args[@]}" --query-gpu=index,clocks.current.graphics,clocks.current.memory,power.draw --format=csv
echo
echo "Idle stays low; under CUDA load graphics should hold at ${GRAPHICS_MHZ} MHz."
echo "Video clocks are not independently configurable and may differ by ~1 boost step."
