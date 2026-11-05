#!/usr/bin/env python3
"""
Convert JSONL benchmark results into pandas DataFrames and generate paper comparison figures.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure utils can be imported regardless of execution working directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import utils

BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def jsonl_to_pkl(jsonl_path: Path, accl: str = "a1000", date_str: str = "26_11_25"):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"Skipping corrupt line: {e}")

    if not records:
        print(f"No records found in {jsonl_path}")
        return None, None

    df_all = pd.DataFrame(records)
    print(f"Loaded {len(df_all)} records from {jsonl_path}")

    # Split into mpi and mpp
    results = {}
    for sys_type in ["mpi", "mpp"]:
        df_sub = df_all[df_all["sys_type"] == sys_type].copy()
        if len(df_sub) == 0:
            continue

        exp_name = df_sub["exp_name"].iloc[0]
        # Drop internal runner fields
        drop_cols = ["sys_type", "exp_name", "trial", "timestamp"]
        df_clean = df_sub.drop(columns=[c for c in drop_cols if c in df_sub.columns])

        pkl_name = f"tb_{exp_name}_stencil_5s_{sys_type}_{accl}_{date_str}.pkl"
        out_pkl = DATA_DIR / pkl_name

        data_dict = {
            "config": {},
            "metadata": {
                "hostname": "node11",
                "start_ts": pd.Timestamp(df_sub["timestamp"].min()),
                "end_ts": pd.Timestamp(df_sub["timestamp"].max()),
                "hosts": "node11,node12,node13,node14,node15,node16,node17,node18,node19,node20",
            },
            "dataframe": df_clean,
        }

        with open(out_pkl, "wb") as fp:
            pickle.dump(data_dict, fp)
        print(f"Saved {sys_type} pkl: {out_pkl} ({len(df_clean)} rows)")
        results[sys_type] = out_pkl

    return results

def generate_paper_figures(accl: str = "a1000", date_str: str = "26_11_25"):
    """Replicates the plots from paper.ipynb, comparing A1000 against H100 and MI300A."""
    print("Generating comparison plots...")
    devices_per_worker = 1
    
    # 1. Load A1000 datasets
    mpi_a1000_pkl = DATA_DIR / f"tb_wscaling_stencil_5s_mpi_{accl}_{date_str}.pkl"
    mpp_a1000_pkl = DATA_DIR / f"tb_wscaling_stencil_5s_mpp_{accl}_{date_str}.pkl"

    benchs_to_load = []
    if mpi_a1000_pkl.exists():
        benchs_to_load.append(mpi_a1000_pkl)
    if mpp_a1000_pkl.exists():
        benchs_to_load.append(mpp_a1000_pkl)
        
    # Also load original paper datasets for comparison
    h100_benchs = [
        DATA_DIR / "tb_wscaling_stencil_5s_mpp_h100_15_07_25.pkl",
        DATA_DIR / "tb_wscaling_stencil_5s_mpi_h100_15_07_25.pkl",
    ]
    mi300a_benchs = [
        DATA_DIR / "tb_wscaling_stencil_5s_mpp_mi300a_15_07_25.pkl",
        DATA_DIR / "tb_wscaling_stencil_5s_mpi_mi300a_15_07_25.pkl",
    ]

    all_dfs = []
    for p in benchs_to_load + [b for b in h100_benchs + mi300a_benchs if b.exists()]:
        df, _ = utils.load_data(p)
        if "mpi" in p.name:
            df["sys"] = "mpi"
        else:
            df["sys"] = "mpp"

        if "a1000" in p.name:
            df["accl"] = "a1000"
        elif "h100" in p.name:
            df["accl"] = "h100"
        elif "mi300a" in p.name:
            df["accl"] = "mi300a"
        
        df["devices"] = df["workers"] * devices_per_worker
        all_dfs.append(df)

    if not all_dfs:
        print("No datasets found to plot.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.drop(columns=[c for c in ['name', 'kernel', 'version', 'hosts', 'time'] if c in all_df.columns], inplace=True)

    agg_all_df = all_df.groupby(
        by=['sys', 'accl', 'devices', 'type', 'radix', 'output', 'size', 'total_flops', 'total_tasks']
    ).agg(
        runtime_mean=('wall_time', 'mean'),
        runtime_std=('wall_time', 'std')
    ).reset_index()

    agg_all_df['tflops_per_sec'] = agg_all_df['total_flops'] * np.pow(1/10, 12) / agg_all_df['runtime_mean']
    agg_all_df['flops_per_task'] = agg_all_df['total_flops'] / agg_all_df['total_tasks']

    agg_all_df_max_flops = agg_all_df.groupby(
        by=['accl', 'devices', 'type', 'radix', 'output'], as_index=False
    )['tflops_per_sec'].max()
    agg_all_df_max_flops.rename(columns={'tflops_per_sec': 'max_tflops_per_sec'}, inplace=True)

    agg_all_df = agg_all_df.merge(agg_all_df_max_flops, on=['accl', 'devices', 'type', 'radix', 'output'])
    agg_all_df['efficiency'] = agg_all_df['tflops_per_sec'] * 100 / agg_all_df['max_tflops_per_sec']
    agg_all_df['task_granularity'] = agg_all_df['runtime_mean'] * agg_all_df['devices'] * np.pow(10, 3) / agg_all_df['total_tasks']
    agg_all_df['config'] = agg_all_df['sys'].str.upper() + "-" + agg_all_df['accl'].str.upper()

    # Calculate METG(50%)
    metg_target = 'task_granularity'
    metg = lambda x: utils.find_metg(x, x_data=metg_target, y_data='efficiency', method=2)

    metg_df = agg_all_df[['config', 'sys', 'accl', 'devices', metg_target, 'efficiency']].groupby(
        ['config', 'sys', 'accl', 'devices']
    ).apply(metg, include_groups=False).reset_index()

    print("=== METG(50%) Results Table ===")
    print(metg_df.to_string())

    # Plot 1: Granularity vs Efficiency (10 devices)
    for num_devices in [1, 2, 5, 10]:
        sub = agg_all_df[agg_all_df['devices'] == num_devices]
        if len(sub) == 0:
            continue
        plot_meta = utils.PlotMeta(
            x_axis='task_granularity',
            y_axis='efficiency',
            hue='config',
            style='config',
            xlabel='Task Granularity (ms)',
            ylabel='Efficiency (%)',
            xticks=[],
            yticks=[],
            legend_title='Environment',
            legend_labels=[],
        )
        utils.generic_line_plot_err_bar(
            dataset=sub,
            title=f'Weak Scaling Efficiency vs Task Granularity ({num_devices} Devices)',
            output_file=FIGURES_DIR / f'a1000_granularity_metg_devices_{num_devices}.pdf',
            plot_meta=plot_meta,
            dashes=False,
            fifty_percent_line=True,
            log_scale=True,
        )
        print(f"Generated {FIGURES_DIR / f'a1000_granularity_metg_devices_{num_devices}.pdf'}")

    # Plot 2: METG(50%) vs Num Devices
    plot_meta_metg = utils.PlotMeta(
        x_axis='devices',
        y_axis='metg',
        hue='config',
        style='config',
        xlabel='Number of Devices',
        ylabel='METG(50%) (ms)',
        xticks=metg_df['devices'].astype('str').unique(),
        yticks=[],
        legend_title='Environment',
        legend_labels=[],
    )
    utils.generic_line_plot_err_bar(
        dataset=metg_df,
        title='Minimum Effective Task Granularity (50%) vs Devices',
        output_file=FIGURES_DIR / 'a1000_metg_vs_devices.pdf',
        plot_meta=plot_meta_metg,
        dashes=False,
        fifty_percent_line=False,
        log_scale_y=True,
        outside_legend=True,
    )
    print(f"Generated {FIGURES_DIR / 'a1000_metg_vs_devices.pdf'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DATA_DIR / "tb_wscaling_stencil_5s.jsonl")
    parser.add_argument("--accl", default="a1000")
    parser.add_argument("--date", default=datetime.now().strftime("%d_%m_%y"))
    args = parser.parse_args()

    if args.jsonl.exists():
        jsonl_to_pkl(args.jsonl, accl=args.accl, date_str=args.date)
    generate_paper_figures(accl=args.accl, date_str=args.date)
