#!/usr/bin/env python3
"""
Rerun Outliers Script for SSCAD 2025 Reproduction
Re-executes benchmark trials where wall_time exceeded a threshold (e.g. > 30s)
and replaces them in tb_wscaling_stencil_5s.jsonl.
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

HOSTS = [f"node{i}" for i in range(11, 21)]

def parse_output(stdout_text):
    data = {}
    for line in stdout_text.splitlines():
        if "Total Tasks" in line:
            m = re.search(r"Total Tasks\s+(\d+)", line)
            if m: data["total_tasks"] = int(m.group(1))
        elif "Total Dependencies" in line:
            m = re.search(r"Total Dependencies\s+(\d+)", line)
            if m: data["total_deps"] = int(m.group(1))
        elif "Total FLOPs" in line:
            m = re.search(r"Total FLOPs\s+(\d+)", line)
            if m: data["total_flops"] = int(m.group(1))
        elif "Total Bytes" in line:
            m = re.search(r"Total Bytes\s+(\d+)", line)
            if m: data["total_bytes"] = int(m.group(1))
        elif "Elapsed Time" in line:
            m = re.search(r"Elapsed Time\s+([\d\.e\+\-]+)\s+seconds", line)
            if m: data["wall_time"] = float(m.group(1))
    return data

def build_command(sys_type, workers, size, steps=20, radix=5, output=4, nblocks=1024, tpblock=512):
    output_bytes = 2 ** output
    iterations = 2 ** (19 - size)
    width = workers
    
    if sys_type == "mpi":
        hosts_list = ",".join([f"{h}:1" for h in HOSTS[:workers]])
        cmd = [
            "/opt/mpich/bin/mpiexec",
            "-wdir", "/tmp",
            "-hosts", hosts_list,
            "-n", str(workers),
            "-env", "LD_LIBRARY_PATH", "/opt/llvm-mpp/lib:/opt/mpich/lib:/opt/ucx/lib",
            "-env", "UCX_TLS", "tcp,sm",
            "-env", "OMPTARGET_DISABLE_MPI_PLUGIN", "1",
            "/opt/llvm-mpp/bin/tb_mpi",
            "-steps", str(steps),
            "-width", str(width),
            "-type", "stencil_1d",
            "-radix", str(radix),
            "-output", str(output_bytes),
            "-kernel", "compute_bound_gpu",
            "-iter", str(iterations),
            "-nblocks", str(nblocks),
            "-tpblock", str(tpblock),
        ]
        version = "bin/tb_mpi_a1000"
        hosts_subset = ",".join(HOSTS[:workers])
    elif sys_type == "mpp":
        proxy_hosts = [f"{h}:1" for h in HOSTS[:workers]]
        all_hosts = ",".join(proxy_hosts + ["node11:1"])
        cmd = [
            "/opt/mpich/bin/mpiexec",
            "-wdir", "/tmp",
            "-hosts", all_hosts,
            "-env", "LD_LIBRARY_PATH", "/opt/llvm-mpp/lib:/opt/mpich/lib:/opt/ucx/lib",
            "-env", "UCX_TLS", "tcp,sm",
            "-n", str(workers),
            "/opt/llvm-mpp/bin/llvm-offload-mpi-proxy-device",
            ":",
            "-wdir", "/tmp",
            "-env", "LD_LIBRARY_PATH", "/opt/llvm-mpp/lib:/opt/mpich/lib:/opt/ucx/lib",
            "-env", "UCX_TLS", "tcp,sm",
            "-env", "CUDA_VISIBLE_DEVICES", "",
            "-n", "1",
            "/opt/llvm-mpp/bin/tb_mpp",
            "-steps", str(steps),
            "-width", str(width),
            "-type", "stencil_1d",
            "-radix", str(radix),
            "-output", str(output_bytes),
            "-kernel", "compute_bound_gpu",
            "-iter", str(iterations),
            "-nblocks", str(nblocks),
            "-tpblock", str(tpblock),
            "-worker", str(workers),
        ]
        version = "bin/tb_mpp_a1000"
        hosts_subset = ",".join(HOSTS[:workers])
    else:
        raise ValueError(f"Unknown sys_type: {sys_type}")
        
    return cmd, version, hosts_subset

def main():
    parser = argparse.ArgumentParser(description="Rerun Outliers")
    parser.add_argument("--jsonl", default="results/tb_wscaling_stencil_5s.jsonl", help="Path to jsonl file")
    parser.add_argument("--threshold", type=float, default=30.0, help="Wall time threshold in seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per run")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist!")
        sys.exit(1)

    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {jsonl_path}")

    # Find outliers
    outliers_idx = [i for i, r in enumerate(records) if r.get("wall_time", 0.0) > args.threshold]
    print(f"Found {len(outliers_idx)} records with wall_time > {args.threshold}s")

    if not outliers_idx:
        print("No outliers to rerun!")
        return

    print("=" * 80)
    print(f"{'Idx':<4} {'Sys':<5} {'W':<3} {'Size':<5} {'Trial':<6} {'Old Wall Time':<15} {'Status':<15}")
    print("=" * 80)
    for i in outliers_idx:
        r = records[i]
        print(f"{i:<4} {r['sys_type']:<5} {r['workers']:<3} {r['size']:<5} {r['trial']:<6} {r['wall_time']:<15.4f}")
    print("=" * 80)

    # Rerun each outlier
    success_count = 0
    for idx_num, i in enumerate(outliers_idx, 1):
        r = records[i]
        sys_type = r["sys_type"]
        workers = r["workers"]
        size = r["size"]
        trial = r["trial"]
        old_wall_time = r["wall_time"]

        print(f"\n[{idx_num}/{len(outliers_idx)}] Rerunning {sys_type.upper()} workers={workers} size={size} trial={trial} (old={old_wall_time:.4f}s)...")
        cmd, version, hosts_subset = build_command(sys_type, workers, size)

        t0 = time.time()
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=args.timeout)
            t1 = time.time()
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] after {args.timeout}s!")
            continue

        parsed = parse_output(res.stdout)
        if "wall_time" not in parsed:
            print(f"  [ERROR]: Output did not contain valid wall_time! Retcode={res.returncode}")
            print(f"  STDOUT: {res.stdout[:300]}")
            print(f"  STDERR: {res.stderr[:300]}")
            continue

        new_wall_time = parsed["wall_time"]
        speedup = old_wall_time / new_wall_time if new_wall_time > 0 else 1.0

        # Update record
        r["wall_time"] = new_wall_time
        r["total_flops"] = parsed.get("total_flops", r["total_flops"])
        r["total_tasks"] = parsed.get("total_tasks", r["total_tasks"])
        r["total_deps"] = parsed.get("total_deps", r["total_deps"])
        r["total_bytes"] = parsed.get("total_bytes", r["total_bytes"])
        r["time"] = t1 - t0
        r["timestamp"] = datetime.now().isoformat()
        r["rerun_from_old_wall_time"] = old_wall_time

        print(f"  -> SUCCESS: new_wall_time={new_wall_time:.6f}s (Speedup={speedup:.2f}x) in {t1 - t0:.2f}s")
        success_count += 1

        # Write progress immediately to file
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 80)
    print(f"Rerun complete: {success_count}/{len(outliers_idx)} outliers successfully updated in {jsonl_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
