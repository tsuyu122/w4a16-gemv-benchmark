#!/usr/bin/env python3
import subprocess
import sys
import os
import json
import shutil
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd, cwd=ROOT_DIR):
    return subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    config_file = os.path.join(ROOT_DIR, "baseline_config.json")
    history_file = os.path.join(ROOT_DIR, "rounds_history.json")

    config = load_json(config_file, {
        "baseline_latency_us": 4535.60,
        "hardware": "12th Gen Intel Core i5-12400F",
        "matrix": {"K": 4096, "N": 4096, "group_size": 128}
    })
    baseline_us = config.get("baseline_latency_us", 4535.60)
    history = load_json(history_file, {"rounds": []})

    current_round = len(history.get("rounds", [])) + 1

    print("=" * 68)
    print("  AI INFERENCE LOW-LEVEL BENCHMARK EVALUATOR")
    print("  Target  : W4A16 Quantized GEMV (LLM Decoding Engine)")
    print(f"  Hardware: {config.get('hardware', 'Local CPU')}")
    print(f"  BASELINE: {baseline_us:.2f} us (Target to beat)")
    print(f"  EXECUTION ROUND: {current_round} of 3")
    print("=" * 68)

    # 1. Compilation
    print(f"\n[Round {current_round} - Step 1] Compiling project...")
    make_cmd = "make" if shutil.which("make") else f"{os.path.expanduser('~')}/.tools/w64devkit/bin/make.exe"
    
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

    # 2. Execution with fixed baseline passed
    print(f"\n[Round {current_round} - Step 2] Running verification and benchmark...")
    bench_exe = os.path.join(ROOT_DIR, "w4a16_bench.exe")
    if not os.path.exists(bench_exe):
        bench_exe = os.path.join(ROOT_DIR, "w4a16_bench")

    bench_run = subprocess.run([bench_exe, "--baseline", str(baseline_us)], cwd=ROOT_DIR)
    if bench_run.returncode != 0:
        print(f"\n[-] Round {current_round} failed: Verification failed or crash occurred.")
        sys.exit(bench_run.returncode)

    # 3. Read scorecard and update history
    result_file = os.path.join(ROOT_DIR, "benchmark_result.json")
    res_data = load_json(result_file, {})

    cand_mean_us = res_data.get("candidate_latency_us", {}).get("mean", 0.0)
    speedup = res_data.get("speedup", 0.0)
    bandwidth = res_data.get("bandwidth_gbps", 0.0)
    max_error = res_data.get("max_abs_error", 0.0)

    round_record = {
        "round": current_round,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": res_data.get("status", "PASS"),
        "max_abs_error": max_error,
        "candidate_latency_us": cand_mean_us,
        "speedup": speedup,
        "bandwidth_gbps": bandwidth
    }
    history["rounds"].append(round_record)
    save_json(history_file, history)

    print("\n" + "=" * 68)
    print(f"  ROUND {current_round} COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print(f"  Baseline Target Latency : {baseline_us:.2f} us")
    print(f"  Your Candidate Latency  : {cand_mean_us:.2f} us")
    print(f"  Your Speedup Factor     : {speedup:.2f}x")
    print(f"  Memory Bandwidth        : {bandwidth:.2f} GB/s")
    print(f"  Max Numerical Error     : {max_error:.6f}")
    print("-" * 68)
    print("  ROUND SUMMARY SO FAR:")
    for r in history["rounds"]:
        print(f"   * Round {r['round']}: {r['candidate_latency_us']:.2f} us ({r['speedup']:.2f}x speedup) [{r['status']}]")
    print("=" * 68)

    if current_round >= 3:
        print("\n[!] Maximum evaluation rounds (3/3) reached for this run.")
        print("    Run 'python reset.py' to reset the repository for the next model.")

if __name__ == "__main__":
    main()
