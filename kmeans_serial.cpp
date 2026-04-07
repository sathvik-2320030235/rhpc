#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace std;

// Configuration
int N = 1000000;
int D = 100;
int K = 100;
int MAX_ITERS = 50;

// Distance function
float distance_sq(const float *p1, const float *p2, int d) {
    float sum = 0;
    for (int i = 0; i < d; ++i) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char** argv) {
    // Read inputs
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) MAX_ITERS = atoi(argv[4]);

    cout << "--- Serial K-Means ---" << endl;
    cout << "N=" << N << " D=" << D << " K=" << K << endl;

    vector<float> points(N * D);
    vector<float> centroids(K * D);
    vector<int> assignments(N, 0);

    // Generate random data
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * D; ++i) {
        points[i] = dist(rng);
    }

    // Initialize centroids
    for (int i = 0; i < K * D; ++i) {
        centroids[i] = points[i];
    }

    auto start_time = chrono::high_resolution_clock::now();

    // K-Means iterations
    for (int iter = 0; iter < MAX_ITERS; ++iter) {

        float total_sse = 0.0f;
        int changes = 0;

        // --- Assignment Step ---
        for (int i = 0; i < N; ++i) {
            float min_dist = 1e30f;
            int best_k = -1;

            for (int j = 0; j < K; ++j) {
                float d = distance_sq(&points[i * D], &centroids[j * D], D);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = j;
                }
            }

            if (assignments[i] != best_k) {
                changes++;
                assignments[i] = best_k;
            }

            total_sse += min_dist;
        }

        // --- Update Step ---
        vector<float> new_centroids(K * D, 0.0f);
        vector<int> counts(K, 0);

        for (int i = 0; i < N; ++i) {
            int cluster = assignments[i];
            counts[cluster]++;

            for (int d = 0; d < D; ++d) {
                new_centroids[cluster * D + d] += points[i * D + d];
            }
        }

        for (int j = 0; j < K; ++j) {
            if (counts[j] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids[j * D + d] =
                        new_centroids[j * D + d] / counts[j];
                }
            }
        }

        cout << "Iter " << iter
             << " | SSE: " << total_sse
             << " | Changes: " << changes << endl;

        if (changes == 0) {
            cout << "Converged early!" << endl;
            break;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "Total computation time: " << diff.count() << " s" << endl;

    return 0;
}