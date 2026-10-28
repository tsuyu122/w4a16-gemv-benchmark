#!/usr/bin/env python3
import subprocess
import sys
import os
import json
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd, cwd=ROOT_DIR):
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res

def main():
    print("=" * 65)
    print("  AI INFERENCE LOW-LEVEL BENCHMARK EVALUATOR")
    print("  Target: W4A16 Quantized GEMV (LLM Decoding Engine)")
    print("=" * 65)

    # 1. Compilation
    print("\n[Stage 1] Compiling project...")
    make_cmd = "make" if shutil.which("make") else f"{os.path.expanduser('~')}/.tools/w64devkit/bin/make.exe"
    
    # Try make, fallback to direct g++ compile if make isn't found
    res = run_command(f"{make_cmd} clean && {make_cmd}")
    if res.returncode != 0:
        print("[!] Make failed, trying direct g++ invocation...")
        gxx_cmd = "g++" if shutil.which("g++") else f"{os.path.expanduser('~')}/.tools/w64devkit/bin/g++.exe"
        compile_cmd = (
            f"{gxx_cmd} -O3 -mavx2 -mfma -fopenmp -std=c++20 -Iinclude "
            f"src/reference.cpp src/candidate.cpp bench/harness.cpp -o w4a16_bench.exe"
        )
        res = run_command(compile_cmd)
        if res.returncode != 0:
            print("[-] COMPILATION FAILED!")
            print(res.stderr)
            sys.exit(1)

    print("[+] Compilation successful: w4a16_bench.exe generated.")

    # 2. Benchmark Execution
    print("\n[Stage 2] Running correctness verification and speed benchmark...")
    bench_exe = os.path.join(ROOT_DIR, "w4a16_bench.exe")
    if not os.path.exists(bench_exe):
        bench_exe = os.path.join(ROOT_DIR, "w4a16_bench")

    bench_run = subprocess.run([bench_exe], cwd=ROOT_DIR)
    if bench_run.returncode != 0:
        print("\n[-] Benchmark execution failed (verification failure or crash)!")
        sys.exit(bench_run.returncode)

    # 3. Read and format JSON results
    json_file = os.path.join(ROOT_DIR, "benchmark_result.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        print("\n" + "=" * 65)
        print("  FINAL EVALUATION SCORECARD")
        print("=" * 65)
        print(f"  Status                  : {data.get('status')}")
        print(f"  Max Absolute Error      : {data.get('max_abs_error'):.6f}")
        print(f"  Baseline Mean Latency   : {data.get('baseline_latency_us'):.2f} us")
        print(f"  Candidate Mean Latency  : {data.get('candidate_latency_us', {}).get('mean', 0.0):.2f} us")
        print(f"  Speedup Factor          : {data.get('speedup'):.2f}x")
        print(f"  Memory Bandwidth        : {data.get('bandwidth_gbps'):.2f} GB/s")
        print("=" * 65)

if __name__ == "__main__":
    main()
