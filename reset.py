#!/usr/bin/env python3
import os
import shutil
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print("  RESETTING BENCHMARK FOR NEXT MODEL")
    print("=" * 65)

    # 1. Restore candidate.cpp
    backup_file = os.path.join(ROOT_DIR, "src", "candidate_initial.cpp")
    target_file = os.path.join(ROOT_DIR, "src", "candidate.cpp")

    if os.path.exists(backup_file):
        shutil.copyfile(backup_file, target_file)
        print("[+] Restored src/candidate.cpp to original baseline template.")
    else:
        print("[!] Warning: src/candidate_initial.cpp not found.")

    # 2. Remove history and results
    files_to_remove = [
        os.path.join(ROOT_DIR, "rounds_history.json"),
        os.path.join(ROOT_DIR, "benchmark_result.json"),
        os.path.join(ROOT_DIR, "w4a16_bench.exe"),
        os.path.join(ROOT_DIR, "w4a16_bench"),
    ]

    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"[+] Removed: {os.path.basename(f)}")
            except Exception as e:
                print(f"[!] Could not remove {f}: {e}")

    # 3. Clean object files
    src_dir = os.path.join(ROOT_DIR, "src")
    bench_dir = os.path.join(ROOT_DIR, "bench")
    for d in [src_dir, bench_dir, ROOT_DIR]:
        if os.path.exists(d):
            for item in os.listdir(d):
                if item.endswith(".o"):
                    os.remove(os.path.join(d, item))

    print("=" * 65)
    print("  REPOSITORY IS FRESH AND READY FOR NEXT TEST RUN")
    print("=" * 65)

if __name__ == "__main__":
    main()
