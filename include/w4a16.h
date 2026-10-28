#pragma once

#include <cstdint>
#include <cstddef>

// Default benchmark matrix dimensions (LLM Decoder Projection)
constexpr int DEFAULT_K = 4096;  // Input dimension (hidden_dim)
constexpr int DEFAULT_N = 4096;  // Output dimension (vocab/intermediate/heads)
constexpr int GROUP_SIZE = 128;  // Quantization block size (AWQ / GPTQ standard)

/**
 * W4A16 Quantized GEMV Contract:
 * 
 * Computes: y = x * W
 * where:
 *   x       : Input activation vector of shape [1, K], 32-bit float
 *   qweight : Quantized weight matrix of shape [K, N], packed as 4-bit nibbles
 *             Memory layout: Column-Major!
 *             Column n has (K / 2) bytes located at qweight + n * (K / 2).
 *             Byte b in column n contains:
 *               - Low 4 bits  (byte & 0x0F)      : weight at row 2*b
 *               - High 4 bits ((byte >> 4) & 0x0F): weight at row 2*b + 1
 * 
 *   scales  : Scale factor per group of 128 weights, shape [N, K / 128]
 *             Scale for column n, group g is at scales[n * (K / 128) + g]
 * 
 *   zeros   : Zero-point offset per group of 128 weights, shape [N, K / 128]
 *             Zero for column n, group g is at zeros[n * (K / 128) + g]
 * 
 *   y       : Output vector of shape [1, N], 32-bit float
 * 
 * Mathematical formula for weight dequantization:
 *   w[k, n] = (q[k, n] - zero) * scale
 */

// Ground-truth reference implementation (baseline oracle)
void reference_w4a16_gemv(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ y,
    int K,
    int N
);

// Candidate implementation (THIS IS WHAT THE MODEL MUST OPTIMIZE)
void candidate_w4a16_gemv(
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ y,
    int K,
    int N
);
