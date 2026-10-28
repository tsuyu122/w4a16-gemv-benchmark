#include "w4a16.h"

/**
 * Reference W4A16 GEMV implementation.
 * 
 * Scalar, single-threaded baseline.
 * Used as the ground-truth oracle for correctness checks.
 */
void reference_w4a16_gemv(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ y,
    int K,
    int N
) {
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
