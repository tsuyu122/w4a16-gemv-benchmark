# W4A16 GEMV AI Inference Benchmark

> **Hard Systems Benchmark for AI Coding Agents**: Low-level C++ / SIMD optimization of the LLM decoding phase (Memory-Bandwidth Bound W4A16 Quantized GEMV).

---

## 🎯 The Benchmark Objective

In modern LLM inference (e.g. *LLaMA*, *Mistral*, *DeepSeek*, *Qwen*), generating tokens one-by-one (**Decoding Phase**) is strictly **memory-bandwidth bound**. Unlike the prompt prefill phase (which is compute-bound matrix multiplication $M > 1$), decoding runs with $M = 1$ vector-matrix multiplication (**GEMV**):

$$y = x \cdot W$$

To maximize token generation speed, inference engines (like *llama.cpp*, *AWQ*, *ExLlamaV2*, *Marlin*) store model weights compressed in **4-bit integers (INT4)** and stream them from RAM/VRAM into CPU/GPU registers, dequantizing them *on-the-fly*.

### Your Challenge:
You are given an unoptimized, scalar baseline in `src/candidate.cpp`.
Your goal is to optimize `candidate_w4a16_gemv` to achieve the **maximum speedup** over the baseline while maintaining **100% numerical precision** ($|y_{candidate} - y_{ref}| \le 1e-3$).

---

## 🔄 The 3-Pass Iterative Evaluation Loop

This benchmark is designed to test how an AI model or human engineer iteratively profiles, debugs, and pushes low-level hardware performance:

| Pass | Goal | What to focus on |
| :--- | :--- | :--- |
| **Pass 1: Functional Acceleration** | Beat the naive baseline | Multi-threading across columns (`#pragma omp parallel for`) and basic SIMD vectorization. Verify that output precision passes. |
| **Pass 2: Architecture Refactoring** | Register-level optimization | Eliminate register spills. Unpack 4-bit nibbles using bitwise shift/and tricks in AVX2 registers. Prevent cache thrashing. |
| **Pass 3: Hardware Roofline Push** | Micro-architectural saturation | Fused Multiply-Add (`_mm256_fmadd_ps`), loop unrolling, memory prefetching (`_mm_prefetch`), and streaming loads. |

---

## 📐 Data Layout & Quantization Specification

* **Input Activation $x$**: Vector of shape `[1, K]`, standard 32-bit floats (`float*`).
* **Matrix Dimensions**: $K = 4096$ (`in_features`), $N = 4096$ (`out_features`). Total weights: $16,777,216$.
* **Group Size**: $G = 128$ weights per quantization block (32 groups per column).
* **Quantized Weights $W$ (`qweight`)**:
  * Stored in **Column-Major order** (crucial for streaming continuous columns).
  * 2 weights are packed into each byte (`uint8_t`):
    * **Low 4 bits** (`byte & 0x0F`): Weight at row $2b$.
    * **High 4 bits** (`(byte >> 4) & 0x0F`): Weight at row $2b + 1$.
* **Scales & Zeros**:
  * Shape: `[N, K / 128]` floats.
  * Weight dequantization formula:
    $$w_{k, n} = (q_{k, n} - \text{zero}_{g, n}) \times \text{scale}_{g, n}$$
* **Output Vector $y$**: Vector of shape `[1, N]`, 32-bit floats.
* **Alignment**: All memory pointers are guaranteed to be **64-byte aligned**.

---

## 🚀 How to Run the Benchmark

### One-Command Evaluation:
```bash
python evaluate.py
```

### Manual Compilation:
```bash
make clean && make
./w4a16_bench.exe
```

### Custom Flags:
```bash
./w4a16_bench.exe --k 4096 --n 4096 --iters 100 --json benchmark_result.json
```

---

## 📊 Score & Evaluation Metrics

A submission is scored automatically by `bench/harness.cpp`:

$$\text{Final Score} = \begin{cases} 
0, & \text{if verification fails } (\max |y_{\text{cand}} - y_{\text{ref}}| > 1e-3) \\
\frac{\text{Baseline Latency (Mean)}}{\text{Candidate Latency (Mean)}}, & \text{if verification passes}
\end{cases}$$

The harness outputs:
* **Max Absolute Error**: Absolute tolerance check.
* **Baseline Latency ($\mu s$)**: Wall-clock time of the reference implementation.
* **Candidate Latency ($\mu s$)**: p50, p90, min, mean wall-clock time.
* **Memory Bandwidth (GB/s)**: Effective memory streaming rate.
* **Speedup Factor**: Improvement ratio ($x$-factor).
* Machine-readable results saved in `benchmark_result.json`.

---

## 🛠️ Ground Rules
1. **Target File**: You must only modify `src/candidate.cpp`.
2. **Contract**: Do not modify `include/w4a16.h`, `src/reference.cpp`, or `bench/harness.cpp`.
3. **Correctness**: Any result where `max_abs_error > 0.001` is marked as `FAIL` with a score of 0.
