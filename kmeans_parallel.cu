#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdlib>
using namespace std;

// Configuration shared with serial
int N = 1000000;
int D = 100;
int K = 100;
int MAX_ITERS = 50;

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// CUDA kernel to compute distances between points and centroids
__global__ void compute_distances(const float* d_points, const float* d_centroids, float* d_distances, int num_points, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        for (int j = 0; j < K; ++j) {
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = d_points[idx * D + d] - d_centroids[j * D + d];
                dist += diff * diff;
            }
            d_distances[idx * K + j] = dist;
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) MAX_ITERS = atoi(argv[4]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Try to use CUDA if available; otherwise gracefully fall back to CPU-only.
    bool use_cuda = true;
    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            std::cerr << "No CUDA devices found or cudaGetDeviceCount failed; running CPU-only (MPI + OpenMP)." << std::endl;
        }
        use_cuda = false;
    }

    if (use_cuda) {
        int deviceId = rank % deviceCount;
        checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");
    }

    int num_local_points = N / size;

    vector<float> local_points(num_local_points * D);
    vector<float> centroids(K * D);
    vector<int> local_assignments(num_local_points, 0);

    // Initial rank prints parameters
    if (rank == 0) {
        cout << "--- Parallel K-Means (MPI + OpenMP + CUDA) ---" << endl;
        cout << "N=" << N << " D=" << D << " K=" << K << " Ranks=" << size << endl;
        
        vector<float> all_points(N * D);
        mt19937 rng(42);
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < N * D; ++i) {
            all_points[i] = dist(rng);
        }

        // Initialize centroids
        for (int i = 0; i < K * D; ++i) {
            centroids[i] = all_points[i];
        }

        MPI_Scatter(all_points.data(), num_local_points * D, MPI_FLOAT, local_points.data(), num_local_points * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, num_local_points * D, MPI_FLOAT, local_points.data(), num_local_points * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(centroids.data(), K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // CUDA allocate memory (only if using CUDA)
    float *d_points = nullptr, *d_centroids = nullptr, *d_distances = nullptr;
    if (use_cuda) {
        checkCuda(cudaMalloc(&d_points, num_local_points * D * sizeof(float)), "cudaMalloc d_points");
        checkCuda(cudaMalloc(&d_centroids, K * D * sizeof(float)), "cudaMalloc d_centroids");
        checkCuda(cudaMalloc(&d_distances, num_local_points * K * sizeof(float)), "cudaMalloc d_distances");

        checkCuda(cudaMemcpy(d_points, local_points.data(), num_local_points * D * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D points");
    }

    vector<float> host_distances(num_local_points * K);

    int num_threads = omp_get_max_threads();
    if (rank == 0) { cout << "Max OpenMP threads per rank: " << num_threads << endl; }

    int blockSize = 256;
    int numBlocks = (num_local_points + blockSize - 1) / blockSize;

    double total_cuda_time = 0;
    double total_omp_time = 0;
    double total_mpi_time = 0;

    auto global_start_time = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        // --- 1. DISTANCE CALCULATION (CUDA if available, otherwise CPU) ---
        auto t0 = MPI_Wtime();
        if (use_cuda) {
            checkCuda(cudaMemcpy(d_centroids, centroids.data(), K * D * sizeof(float), cudaMemcpyHostToDevice),
                      "cudaMemcpy H2D centroids");
            compute_distances<<<numBlocks, blockSize>>>(d_points, d_centroids, d_distances, num_local_points, D, K);
            checkCuda(cudaGetLastError(), "compute_distances kernel launch");
            checkCuda(cudaDeviceSynchronize(), "compute_distances sync");
            checkCuda(cudaMemcpy(host_distances.data(), d_distances, num_local_points * K * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy D2H distances");
        } else {
            // Pure CPU distance computation: fill host_distances directly.
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_local_points; ++i) {
                for (int j = 0; j < K; ++j) {
                    float dist = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        float diff = local_points[i * D + d] - centroids[j * D + d];
                        dist += diff * diff;
                    }
                    host_distances[i * K + j] = dist;
                }
            }
        }
        auto t1 = MPI_Wtime();
        total_cuda_time += (t1 - t0);

        // --- 2. OPENMP ASSIGNMENT AND LOCAL SUMS ---
        float local_sse = 0.0f;
        int local_changes = 0;
        
        vector<float> new_centroids_local(K * D, 0.0f);
        vector<int> counts_local(K, 0);

        auto t2 = MPI_Wtime();
        
        #pragma omp parallel
        {
            float thread_sse = 0.0f;
            int thread_changes = 0;
            vector<float> thread_centroids(K * D, 0.0f);
            vector<int> thread_counts(K, 0);

            #pragma omp for
            for (int i = 0; i < num_local_points; ++i) {
                float min_dist = 1e30f;
                int best_k = -1;
                for (int j = 0; j < K; ++j) {
                    float dist = host_distances[i * K + j];
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = j;
                    }
                }
                
                if (local_assignments[i] != best_k) {
                    thread_changes++;
                    local_assignments[i] = best_k;
                }
                thread_sse += min_dist;
                
                thread_counts[best_k]++;
                for(int d = 0; d < D; ++d) {
                    thread_centroids[best_k * D + d] += local_points[i * D + d];
                }
            }

            #pragma omp critical
            {
                local_sse += thread_sse;
                local_changes += thread_changes;
                for (int j = 0; j < K; ++j) {
                    counts_local[j] += thread_counts[j];
                    for(int d = 0; d < D; ++d) {
                        new_centroids_local[j * D + d] += thread_centroids[j * D + d];
                    }
                }
            }
        }
        
        auto t3 = MPI_Wtime();
        total_omp_time += (t3 - t2);

        // --- 3. MPI GLOBAL REDUCTION ---
        auto t4 = MPI_Wtime();
        int global_changes = 0;
        float global_sse = 0.0f;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        vector<float> new_centroids_global(K * D, 0.0f);
        vector<int> counts_global(K, 0);

        MPI_Allreduce(new_centroids_local.data(), new_centroids_global.data(), K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(counts_local.data(), counts_global.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Update centroids
        for (int j = 0; j < K; ++j) {
            if (counts_global[j] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids[j * D + d] = new_centroids_global[j * D + d] / counts_global[j];
                }
            }
        }
        auto t5 = MPI_Wtime();
        total_mpi_time += (t5 - t4);

        if (rank == 0) {
            cout << "Iter " << iter << " | SSE: " << global_sse << " | Changes: " << global_changes << endl;
        }

        if (global_changes == 0) break;
    }

    if (use_cuda) {
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_distances);
    }

    auto global_end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "--- Benchmark ---" << endl;
        cout << "Total Runtime:     " << (global_end_time - global_start_time) << " s" << endl;
        cout << "CUDA Dist Time:    " << total_cuda_time << " s" << endl;
        cout << "OpenMP Assign Time:" << total_omp_time << " s" << endl;
        cout << "MPI Reduce Time:   " << total_mpi_time << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}
