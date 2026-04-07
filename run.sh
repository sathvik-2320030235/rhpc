#!/usr/bin/env bash
set -euo pipefail

N="${1:-1000000}"
D="${2:-100}"
K="${3:-100}"
ITERS="${4:-50}"

echo "--- Serial ---"
./kmeans_serial "$N" "$D" "$K" "$ITERS"

echo
echo "--- OpenMP ---"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
./kmeans_omp "$N" "$D" "$K" "$ITERS"

if [[ -x "./kmeans_parallel" ]] && command -v mpiexec >/dev/null 2>&1; then
  echo
  echo "--- Hybrid MPI + CUDA ---"
  MPI_RANKS="${MPI_RANKS:-2}"
  mpiexec -n "$MPI_RANKS" ./kmeans_parallel "$N" "$D" "$K" "$ITERS"
fi
