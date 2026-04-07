## rHPC K-Means (Serial / OpenMP / CUDA+MPI)

This folder contains:
- **`kmeans_serial.cpp`**: serial k-means
- **`kmeans_omp.cpp`**: OpenMP k-means (same algorithm/CLI as serial)
- **`kmeans_parallel.cu`**: hybrid MPI + OpenMP + CUDA version (optional toolchain)
- **`kmeans_gui.py`**: small Tkinter GUI to compile/run

## Build (Linux)

Build serial + OpenMP:

```bash
./build.sh
```

Or directly:

```bash
make
```

Optional hybrid build (needs `nvcc` + `mpicxx`):

```bash
make parallel
```

## Run

All binaries accept:

```bash
./kmeans_serial N D K MAX_ITERS
./kmeans_omp    N D K MAX_ITERS
```

Convenience runner:

```bash
./run.sh 1000000 100 100 50
```

Hybrid (if built) example:

```bash
OMP_NUM_THREADS=4 mpiexec -n 2 ./kmeans_parallel 1000000 100 100 50
```

## GUI

```bash
python3 kmeans_gui.py
```

On Linux it uses `./build.sh` and runs `./kmeans_serial` / `./kmeans_parallel` when present.






make
export DISPLAY=:0
python3 bfs_gui.py
mpirun -np 2 ./bfs_baseline 15 16
mpirun -np 2 ./bfs_hybrid 15 16
python3 kmeans_gui.py

./build.sh
./kmeans_serial 2000 8 5 10
OMP_NUM_THREADS=4 ./kmeans_omp 2000 8 5 10

./build.sh              # already OK
export OMP_NUM_THREADS=4
mpiexec -n 2 ./kmeans_parallel 2000 8 5 10
python3 kmeans_gui.py

