#include "w4a16.h"

/**
 * ============================================================================
 * CANDIDATE IMPLEMENTATION: W4A16 Quantized GEMV (Decoding Phase)
 * ============================================================================
 * 
 * YOUR TASK:
 * Maximize the execution speed (minimize latency in microseconds) of this function
 * while maintaining strict numerical precision against reference_w4a16_gemv.
 * 
 * RULES:
 * 1. You may use any SIMD intrinsics (AVX2, FMA, SSE), OpenMP, thread pools,
 *    register unrolling, inline assembly, or cache prefetching.
 * 2. The output vector y[n] must match the reference within tolerance (|diff| < 1e-3).
 * 3. Do not modify the function signature or mutate the input pointers.
 * 4. Memory for x, qweight, scales, zeros, and y is guaranteed to be 64-byte aligned.
 * 
 * HARDWARE HINTS (Intel x86_64 AVX2):
 * - Use _mm256_loadu_si256 or _mm256_load_si256 to read packed 4-bit weights.
 * - Unpack 4-bit nibbles using bitwise shifts and masks:
 *     _mm256_and_si256(raw, mask_0f)
 *     _mm256_srli_epi16(raw, 4)
 * - Convert integers to floats using _mm256_cvtepi32_ps.
 * - Accumulate products using _mm256_fmadd_ps.
 * - Parallelize across columns n with OpenMP: #pragma omp parallel for
 * ============================================================================
 */

void candidate_w4a16_gemv(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ y,
    int K,
    int N
) {
    // -------------------------------------------------------------------------
    // Baseline starting implementation (Unoptimized Scalar)
    // -------------------------------------------------------------------------
    const int num_groups = K / GROUP_SIZE;
    const int bytes_per_col = K / 2;

    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        const uint8_t* col_weights = qweight + n * bytes_per_col;
        const float* col_scales = scales + n * num_groups;
        const float* col_zeros = zeros + n * num_groups;

        for (int g = 0; g < num_groups; ++g) {
            const float s = col_scales[g];
            const float z = col_zeros[g];
            const int group_start_k = g * GROUP_SIZE;
            const uint8_t* group_bytes = col_weights + (g * (GROUP_SIZE / 2));

            for (int b = 0; b < GROUP_SIZE / 2; ++b) {
                const uint8_t byte_val = group_bytes[b];
                const float q0 = static_cast<float>(byte_val & 0x0F);
                const float q1 = static_cast<float>((byte_val >> 4) & 0x0F);

                const float w0 = (q0 - z) * s;
                const float w1 = (q1 - z) * s;

                const int k0 = group_start_k + 2 * b;
                const int k1 = k0 + 1;

                sum += x[k0] * w0 + x[k1] * w1;
            }
        }
        y[n] = sum;
    }
}
