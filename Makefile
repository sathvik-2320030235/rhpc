CXX ?= g++
CXXFLAGS ?= -O3 -std=c++11

.PHONY: all clean serial omp parallel

all: serial omp

serial: kmeans_serial

omp: kmeans_omp

kmeans_serial: kmeans_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

kmeans_omp: kmeans_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

# Optional: requires MPI + CUDA toolchain installed and configured.
parallel: kmeans_parallel

kmeans_parallel: kmeans_parallel.cu
	@command -v nvcc >/dev/null 2>&1 || (echo "nvcc not found; install CUDA or skip 'make parallel'." && exit 1)
	@command -v mpicxx >/dev/null 2>&1 || (echo "mpicxx not found; install an MPI implementation or skip 'make parallel'." && exit 1)
	nvcc -O3 -Xcompiler -fopenmp -ccbin mpicxx $< -o $@

clean:
	rm -f kmeans_serial kmeans_omp kmeans_parallel
