#include "w4a16.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cstdlib>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
static void* aligned_alloc_64(size_t size) {
    return _aligned_malloc(size, 64);
}
static void aligned_free_64(void* ptr) {
    _aligned_free(ptr);
}
#else
static void* aligned_alloc_64(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
    return ptr;
}
static void aligned_free_64(void* ptr) {
    free(ptr);
}
#endif

int main(int argc, char** argv) {
    int K = DEFAULT_K;
    int N = DEFAULT_N;
    int timed_iters = 50;
    int warmup_iters = 10;
    std::string json_path = "benchmark_result.json";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) timed_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
    }

    const int num_groups = K / GROUP_SIZE;
    const size_t bytes_x = K * sizeof(float);
    const size_t bytes_qweight = (static_cast<size_t>(K) / 2) * N;
    const size_t bytes_scales = static_cast<size_t>(num_groups) * N * sizeof(float);
    const size_t bytes_zeros = static_cast<size_t>(num_groups) * N * sizeof(float);
    const size_t bytes_y = N * sizeof(float);

    const double total_data_bytes = static_cast<double>(bytes_x + bytes_qweight + bytes_scales + bytes_zeros + bytes_y);

    float* x = static_cast<float*>(aligned_alloc_64(bytes_x));
    uint8_t* qweight = static_cast<uint8_t*>(aligned_alloc_64(bytes_qweight));
    float* scales = static_cast<float*>(aligned_alloc_64(bytes_scales));
    float* zeros = static_cast<float*>(aligned_alloc_64(bytes_zeros));
    float* y_ref = static_cast<float*>(aligned_alloc_64(bytes_y));
    float* y_cand = static_cast<float*>(aligned_alloc_64(bytes_y));

    if (!x || !qweight || !scales || !zeros || !y_ref || !y_cand) {
        std::cerr << "Error: Memory allocation failed!\n";
        return 1;
    }

    std::cout << "=================================================================\n";
    std::cout << "  AI INFERENCE BENCHMARK: W4A16 GEMV (Decoding Phase)\n";
    std::cout << "=================================================================\n";
    std::cout << "  Matrix Shape  : [K=" << K << ", N=" << N << "]\n";
    std::cout << "  Group Size    : " << GROUP_SIZE << " weights/group\n";
    std::cout << "  Total Weights : " << (static_cast<double>(K) * N / 1e6) << " M weights\n";
    std::cout << "  Weight Memory : " << (static_cast<double>(bytes_qweight) / (1024.0 * 1024.0)) << " MB (INT4 packed)\n";
    std::cout << "-----------------------------------------------------------------\n";

    // Deterministic random data generation
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dist_byte(0, 255);
    std::uniform_real_distribution<float> dist_scale(0.005f, 0.05f);
    std::uniform_real_distribution<float> dist_zero(1.0f, 14.0f);

    for (int i = 0; i < K; ++i) x[i] = dist_x(gen);
    for (size_t i = 0; i < bytes_qweight; ++i) qweight[i] = static_cast<uint8_t>(dist_byte(gen));
    for (size_t i = 0; i < num_groups * N; ++i) {
        scales[i] = dist_scale(gen);
        zeros[i] = dist_zero(gen);
    }
    std::memset(y_ref, 0, bytes_y);
    std::memset(y_cand, 0, bytes_y);

    std::cout << "[1/3] Running Reference Oracle...\n";
    auto ref_t0 = std::chrono::high_resolution_clock::now();
    reference_w4a16_gemv(x, qweight, scales, zeros, y_ref, K, N);
    auto ref_t1 = std::chrono::high_resolution_clock::now();
    double ref_latency_us = std::chrono::duration<double, std::micro>(ref_t1 - ref_t0).count();

    // Additional runs to get a stable baseline reference average
    const int ref_samples = 5;
    double ref_total_us = ref_latency_us;
    for (int i = 1; i < ref_samples; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        reference_w4a16_gemv(x, qweight, scales, zeros, y_ref, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        ref_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    double baseline_mean_us = ref_total_us / ref_samples;
    std::cout << "      Baseline Reference Latency: " << std::fixed << std::setprecision(2) 
              << baseline_mean_us << " us (" << (baseline_mean_us / 1000.0) << " ms)\n";

    std::cout << "[2/3] Verifying Numerical Correctness of Candidate...\n";
    candidate_w4a16_gemv(x, qweight, scales, zeros, y_cand, K, N);

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    int mismatch_idx = -1;

    for (int n = 0; n < N; ++n) {
        float diff = std::abs(y_cand[n] - y_ref[n]);
        float denom = std::abs(y_ref[n]) + 1e-5f;
        float rel = diff / denom;

        if (diff > max_abs_error) {
            max_abs_error = diff;
        }
        if (rel > max_rel_error) {
            max_rel_error = rel;
        }
        if (diff > 1e-3f && mismatch_idx == -1) {
            mismatch_idx = n;
        }
    }

    bool correct = (max_abs_error <= 1e-3f);
    if (!correct) {
        std::cerr << "[-] VERIFICATION FAILED!\n";
        std::cerr << "    Max Absolute Error : " << max_abs_error << " (Tolerance: 1e-3)\n";
        std::cerr << "    Max Relative Error : " << max_rel_error << "\n";
        std::cerr << "    First Mismatch Index: n = " << mismatch_idx << "\n";
        std::cerr << "    Reference value     : " << y_ref[mismatch_idx] << "\n";
        std::cerr << "    Candidate value     : " << y_cand[mismatch_idx] << "\n";

        std::ofstream json_file(json_path);
        json_file << "{\n"
                  << "  \"status\": \"FAIL\",\n"
                  << "  \"error\": \"Numerical precision check failed\",\n"
                  << "  \"max_abs_error\": " << max_abs_error << ",\n"
                  << "  \"tolerance\": 0.001,\n"
                  << "  \"baseline_latency_us\": " << baseline_mean_us << ",\n"
                  << "  \"speedup\": 0.0\n"
                  << "}\n";
        return 1;
    }
    std::cout << "      [PASS] Numerical precision verified! Max abs error: " 
              << std::scientific << std::setprecision(3) << max_abs_error << "\n";

    std::cout << "[3/3] Profiling Candidate Execution Speed...\n";
    std::cout << "      Warming up CPU (" << warmup_iters << " iterations)...\n";
    for (int i = 0; i < warmup_iters; ++i) {
        candidate_w4a16_gemv(x, qweight, scales, zeros, y_cand, K, N);
    }

    std::vector<double> latencies;
    latencies.reserve(timed_iters);

    std::cout << "      Running benchmark (" << timed_iters << " iterations)...\n";
    for (int i = 0; i < timed_iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        candidate_w4a16_gemv(x, qweight, scales, zeros, y_cand, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies.push_back(us);
    }

    std::sort(latencies.begin(), latencies.end());
    double min_us = latencies.front();
    double p50_us = latencies[latencies.size() / 2];
    double p90_us = latencies[static_cast<size_t>(latencies.size() * 0.90)];
    double p99_us = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    double sum_us = 0.0;
    for (double v : latencies) sum_us += v;
    double mean_us = sum_us / latencies.size();

    double speedup = baseline_mean_us / mean_us;
    double bandwidth_gbps = (total_data_bytes / (mean_us * 1e-6)) / 1e9;
    double tflops = (2.0 * static_cast<double>(K) * N) / (mean_us * 1e-6) / 1e12;

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  BENCHMARK RESULTS\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Baseline Latency (Mean) : " << std::fixed << std::setprecision(2) << baseline_mean_us << " us\n";
    std::cout << "  Candidate Latency (Min) : " << min_us << " us\n";
    std::cout << "  Candidate Latency (p50) : " << p50_us << " us\n";
    std::cout << "  Candidate Latency (p90) : " << p90_us << " us\n";
    std::cout << "  Candidate Latency (Mean): " << mean_us << " us (" << (mean_us / 1000.0) << " ms)\n";
    std::cout << "  Memory Bandwidth        : " << bandwidth_gbps << " GB/s\n";
    std::cout << "  Throughput              : " << tflops << " TFLOPs\n";
    std::cout << "  SPEEDUP                 : " << std::setprecision(2) << speedup << "x\n";
    std::cout << "=================================================================\n";

    std::ofstream json_file(json_path);
    json_file << "{\n"
              << "  \"status\": \"PASS\",\n"
              << "  \"matrix\": {\"K\": " << K << ", \"N\": " << N << ", \"group_size\": " << GROUP_SIZE << "},\n"
              << "  \"baseline_latency_us\": " << baseline_mean_us << ",\n"
              << "  \"candidate_latency_us\": {\n"
              << "    \"min\": " << min_us << ",\n"
              << "    \"p50\": " << p50_us << ",\n"
              << "    \"p90\": " << p90_us << ",\n"
              << "    \"mean\": " << mean_us << "\n"
              << "  },\n"
              << "  \"speedup\": " << speedup << ",\n"
              << "  \"bandwidth_gbps\": " << bandwidth_gbps << ",\n"
              << "  \"tflops\": " << tflops << ",\n"
              << "  \"max_abs_error\": " << max_abs_error << "\n"
              << "}\n";

    aligned_free_64(x);
    aligned_free_64(qweight);
    aligned_free_64(scales);
    aligned_free_64(zeros);
    aligned_free_64(y_ref);
    aligned_free_64(y_cand);

    return 0;
}
