#!/usr/bin/env bash
set -euo pipefail

echo "Compiling (serial + OpenMP)..."
make clean >/dev/null 2>&1 || true
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" all

echo "Optional: build CUDA+MPI version (if available)..."
if command -v nvcc >/dev/null 2>&1 && command -v mpicxx >/dev/null 2>&1; then
  make parallel || true
else
  echo "Skipping parallel build (need nvcc + mpicxx)."
fi

echo "Done."
