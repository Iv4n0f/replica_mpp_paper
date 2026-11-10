#!/usr/bin/env python3
"""
Cluster Experiment Runner for SSCAD 2025 Reproduction
Nodes: node11 through node20 (Lenovo ThinkStation P3 Tiny, RTX A1000)
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
from datetime import datetime

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

def build_command(sys_type, workers, size, steps, width, radix, output, nblocks, tpblock):
    output_bytes = 2 ** output
    iterations = 2 ** (19 - size)
    
    if sys_type == "mpi":
        hosts_list = ",".join([f"{h}:1" for h in HOSTS[:workers]])
        cmd = [
            "/opt/mpich/bin/mpiexec",
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
            "-hosts", all_hosts,
            "-env", "LD_LIBRARY_PATH", "/opt/llvm-mpp/lib:/opt/mpich/lib:/opt/ucx/lib",
            "-env", "UCX_TLS", "tcp,sm",
            "-n", str(workers),
            "/opt/llvm-mpp/bin/llvm-offload-mpi-proxy-device",
            ":",
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
        ]
        version = "bin/tb_mpp_a1000"
        hosts_subset = ",".join(HOSTS[:workers])
    else:
        raise ValueError(f"Unknown sys_type: {sys_type}")
        
    return cmd, version, hosts_subset

def run_experiment(exp_name, output_file, sys_types, workers_list, sizes_list, outputs_list, steps, width_fn, radix=5, nblocks=1024, tpblock=512, trials=3, timeout=300):
    # Load completed keys
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        key = (
                            record["sys_type"],
                            record["exp_name"],
                            record["workers"],
                            record["size"],
                            record["output"],
                            record["trial"],
                        )
                        completed.add(key)
                    except Exception:
                        pass
                        
    total_runs = len(sys_types) * len(workers_list) * len(sizes_list) * len(outputs_list) * trials
    done_runs = len(completed)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment '{exp_name}': {done_runs}/{total_runs} already completed. Output: {output_file}")

    with open(output_file, "a") as f_out:
        for sys_type in sys_types:
            for workers in workers_list:
                width = width_fn(workers)
                for size in sizes_list:
                    for output in outputs_list:
                        for trial in range(trials):
                            key = (sys_type, exp_name, workers, size, output, trial)
                            if key in completed:
                                continue
                                
                            cmd, version, hosts_subset = build_command(
                                sys_type=sys_type,
                                workers=workers,
                                size=size,
                                steps=steps,
                                width=width,
                                radix=radix,
                                output=output,
                                nblocks=nblocks,
                                tpblock=tpblock,
                            )
                            
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"[{now_str}] RUN: {sys_type.upper()} | workers={workers} | size={size} | trial={trial+1}/{trials} | width={width}...", flush=True)
                            
                            t0 = time.time()
                            try:
                                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
                                t1 = time.time()
                            except subprocess.TimeoutExpired:
                                print(f"  [TIMEOUT] after {timeout}s: {' '.join(cmd)}", flush=True)
                                continue

                            if res.returncode != 0:
                                print(f"  [ERROR exit {res.returncode}]:\n  STDERR: {res.stderr[:300]}", flush=True)
                                continue

                            parsed = parse_output(res.stdout)
                            if "wall_time" not in parsed:
                                print(f"  [WARNING] No wall_time in output! Skipping.\n  STDOUT: {res.stdout[:200]}", flush=True)
                                continue

                            row = {
                                "sys_type": sys_type,
                                "exp_name": exp_name,
                                "name": "task-bench",
                                "tpblock": tpblock,
                                "nblocks": nblocks,
                                "total_flops": parsed.get("total_flops", 0),
                                "total_tasks": parsed.get("total_tasks", 0),
                                "total_deps": parsed.get("total_deps", 0),
                                "total_bytes": parsed.get("total_bytes", 0),
                                "wall_time": parsed.get("wall_time", 0.0),
                                "radix": radix,
                                "kernel": "compute_bound_gpu",
                                "type": "stencil_1d",
                                "workers": workers,
                                "size": size,
                                "ngraphs": 1,
                                "steps": steps,
                                "output": output,
                                "width": width,
                                "version": version,
                                "hosts": hosts_subset,
                                "time": t1 - t0,
                                "trial": trial,
                                "timestamp": datetime.now().isoformat(),
                            }
                            
                            f_out.write(json.dumps(row) + "\n")
                            f_out.flush()
                            os.fsync(f_out.fileno())
                            completed.add(key)
                            done_runs += 1
                            
                            print(f"  -> wall_time={row['wall_time']:.6f}s (FLOP/s={row['total_flops']/row['wall_time']:.2e}) in {row['time']:.2f}s [{done_runs}/{total_runs}]", flush=True)

def main():
    parser = argparse.ArgumentParser(description="SSCAD 2025 Benchmarks Runner")
    parser.add_argument("--experiment", choices=["weak", "strong", "comm", "all", "smoke"], default="smoke")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--outdir", default=os.path.expanduser("~/results"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    if args.experiment == "smoke":
        # Quick test on 1 and 2 workers, 2 sizes
        output_file = os.path.join(args.outdir, "smoke_test.jsonl")
        run_experiment(
            exp_name="smoke",
            output_file=output_file,
            sys_types=["mpi", "mpp"],
            workers_list=[1, 2],
            sizes_list=[10, 14],
            outputs_list=[4],
            steps=10,
            width_fn=lambda w: w,
            radix=5,
            trials=1,
        )
    elif args.experiment in ["weak", "all"]:
        # Weak Scaling: width = workers, sizes 2..18 (step 2)
        output_file = os.path.join(args.outdir, "tb_wscaling_stencil_5s.jsonl")
        run_experiment(
            exp_name="wscaling",
            output_file=output_file,
            sys_types=["mpi", "mpp"],
            workers_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            sizes_list=[2, 4, 6, 8, 10, 12, 14, 16, 18],
            outputs_list=[4],
            steps=20,
            width_fn=lambda w: w,
            radix=5,
            trials=args.trials,
        )

    if args.experiment in ["strong", "all"]:
        # Strong Scaling: width = 10 (fixed), sizes 2..18 (step 2)
        output_file = os.path.join(args.outdir, "tb_sscaling_stencil_5s.jsonl")
        run_experiment(
            exp_name="sscaling",
            output_file=output_file,
            sys_types=["mpi", "mpp"],
            workers_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            sizes_list=[2, 4, 6, 8, 10, 12, 14, 16, 18],
            outputs_list=[4],
            steps=20,
            width_fn=lambda w: 10,
            radix=5,
            trials=args.trials,
        )

    if args.experiment in ["comm", "all"]:
        # Communication Scaling: 10 workers, width 10, outputs 4..28 (step 4), sizes 2, 6, 10, 14, 18
        output_file = os.path.join(args.outdir, "tb_comm_stencil.jsonl")
        run_experiment(
            exp_name="comm",
            output_file=output_file,
            sys_types=["mpi", "mpp"],
            workers_list=[10],
            sizes_list=[2, 6, 10, 14, 18],
            outputs_list=[4, 8, 12, 16, 20, 24, 28],
            steps=10,
            width_fn=lambda w: 10,
            radix=5,
            trials=args.trials,
        )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished experiment {args.experiment}!")

if __name__ == "__main__":
    main()
