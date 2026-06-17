#!/usr/bin/env python3
"""
Comprehensive experiment runner for SPLA / LAGraph benchmarks.

Covers BFS and SSSP on multiple datasets (directed + undirected), compiles
the LAGraph C runners, then saves raw CSVs to results/ and plots to
presentation/pictures/.

Usage:
    cd lagraph-spla/src
    python run_experiments.py [--algo bfs|sssp|all] [--runs N]
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SRC_DIR  = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent.parent          # graph-exp-ws/
RESULTS  = REPO_DIR / "lagraph-spla" / "results"
PICS     = REPO_DIR / "lagraph-spla" / "presentation" / "pictures"
DATASETS = REPO_DIR / "datasets"

BFS_RESULTS  = RESULTS / "bfs"
SSSP_RESULTS = RESULTS / "sssp"
BFS_PICS     = PICS / "bfs"
SSSP_PICS    = PICS / "sssp"

for d in [BFS_RESULTS, SSSP_RESULTS, BFS_PICS, SSSP_PICS]:
    d.mkdir(parents=True, exist_ok=True)

# Compiled executables (built in-place next to the runner source)
LAGRAPH_BFS  = SRC_DIR / "bfs_runner"
LAGRAPH_SSSP = SRC_DIR / "sssp_runner"

# SPLA executables (pre-built)
# Look first in typical build location, fall back to repo parent
def _find_spla(name: str) -> Path:
    candidates = [
        REPO_DIR.parent / "spla" / "build" / name,
        Path.home() / "spla" / "build" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"SPLA executable '{name}' not found; tried: {candidates}"
    )

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# directed=True means the graph is naturally directed (asymmetric edges)
BFS_DATASETS = [
    ("soc-LiveJournal1.mtx", True,  68_993_773),
    ("patents.mtx",          True,  14_970_767),
    ("wiki-Talk.mtx",        True,   5_021_410),
    ("rgg_n_2_22_s0.mtx",   False, 30_359_198),  # undirected by nature
]

SSSP_DATASETS = [
    ("soc-LiveJournal1.mtx", True),
    ("patents.mtx",          True),
    ("wiki-Talk.mtx",        True),
]

SSSP_DELTA = "1.0"

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def get_omp_prefix() -> str:
    return subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()


def compile_runner(src: Path, out: Path, omp_path: str):
    lagraph_dir = REPO_DIR.parent / "LAGraph"
    cmd = [
        "clang", "-O3", str(src), "-o", str(out),
        "-Xpreprocessor", "-fopenmp",
        "-I", f"{omp_path}/include",
        "-I", str(lagraph_dir / "include"),
        "-I", "/usr/local/include/suitesparse",
        str(lagraph_dir / "build" / "src" / "libLAGraph.a"),
        "-L", f"{omp_path}/lib",
        "-L", "/usr/local/lib",
        "-lgraphblas", "-lomp",
        "-Wl,-rpath,/usr/local/lib",
        f"-Wl,-rpath,{omp_path}/lib",
    ]
    print(f"  Compiling {src.name} → {out.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Compilation failed: {src.name}")
    print("  OK")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[:500]}")
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return r.stdout


def parse_lagraph_bfs_ms(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.startswith("cpu(s):"):
            return float(line.split(":", 1)[1].strip()) * 1000.0
    raise RuntimeError(f"cpu(s) not found in:\n{stdout[:400]}")


def parse_lagraph_sssp_ms(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.startswith("TIME:"):
            return float(line.split(":", 1)[1].strip()) * 1000.0
    raise RuntimeError(f"TIME: not found in:\n{stdout[:400]}")


def parse_spla_gpu_times(stdout: str) -> list:
    for line in stdout.splitlines():
        if line.startswith("gpu(ms):"):
            payload = line.replace("gpu(ms):", "").replace(" ", "").strip().strip(",")
            return [float(x) for x in payload.split(",") if x]
    raise RuntimeError(f"gpu(ms): not found in:\n{stdout[:400]}")


def diag_visited(stdout: str) -> int:
    for line in stdout.splitlines():
        if "DIAG: visited=" in line:
            return int(line.split("visited=")[1].strip())
    return -1

# ---------------------------------------------------------------------------
# BFS experiment
# ---------------------------------------------------------------------------

def run_bfs(runs: int):
    print("\n" + "="*60)
    print("BFS EXPERIMENT")
    print("="*60)

    omp_path = get_omp_prefix()
    compile_runner(SRC_DIR / "bfs_runner.c", LAGRAPH_BFS, omp_path)

    try:
        spla_bfs = _find_spla("bfs")
        has_spla = True
    except FileNotFoundError as e:
        print(f"  Warning: {e} — skipping SPLA runs")
        has_spla = False

    rows = []   # [graph, library_mode, mode, iteration, time_ms, visited]

    for graph_name, is_directed, _ in BFS_DATASETS:
        dataset = DATASETS / graph_name
        if not dataset.exists():
            print(f"  Skip (missing): {graph_name}")
            continue
        print(f"\n  {graph_name} (directed={is_directed})")

        # LAGraph directed (kind=1, sym=0)
        print("    LAGraph directed ...")
        for i in range(1, runs + 1):
            out = run_cmd([str(LAGRAPH_BFS), str(dataset), "1", "0"])
            t   = parse_lagraph_bfs_ms(out)
            v   = diag_visited(out)
            rows.append([graph_name, "LAGraph directed",   "directed",   i, t, v])
            print(f"      run {i}: {t:.1f} ms  visited={v}")

        # LAGraph undirected (kind=0, sym=1)
        print("    LAGraph undirected ...")
        for i in range(1, runs + 1):
            out = run_cmd([str(LAGRAPH_BFS), str(dataset), "0", "1"])
            t   = parse_lagraph_bfs_ms(out)
            v   = diag_visited(out)
            rows.append([graph_name, "LAGraph undirected", "undirected",  i, t, v])
            print(f"      run {i}: {t:.1f} ms  visited={v}")

        if has_spla:
            for label, undirected_flag in [("SpLA directed", "false"), ("SpLA undirected", "true")]:
                print(f"    {label} ...")
                out = run_cmd([
                    str(spla_bfs),
                    f"--mtxpath={dataset}",
                    f"--niters={runs}",
                    "--source=0",
                    "--run-ref=false", "--run-cpu=false", "--run-gpu=true",
                    f"--undirected={undirected_flag}",
                ])
                times = parse_spla_gpu_times(out)
                mode  = "directed" if undirected_flag == "false" else "undirected"
                for i, t in enumerate(times[:runs], start=1):
                    rows.append([graph_name, label, mode, i, t, -1])
                    print(f"      run {i}: {t:.1f} ms")

    # Save CSV
    csv_path = BFS_RESULTS / "bfs_results_v2.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Graph", "LibraryMode", "Mode", "Iteration", "Time_ms", "Visited"])
        w.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    _plot_bfs(rows)
    return rows


def _plot_bfs(rows):
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(rows, columns=["Graph", "LibraryMode", "Mode", "Iteration", "Time_ms", "Visited"])
    df_warm = df[df["Iteration"] > 1].copy()

    order_modes = ["LAGraph directed", "LAGraph undirected", "SpLA directed", "SpLA undirected"]
    name_map = {
        "soc-LiveJournal1.mtx": "LiveJournal",
        "patents.mtx":          "Patents",
        "wiki-Talk.mtx":        "wiki-Talk",
        "rgg_n_2_22_s0.mtx":   "RGG",
    }
    df_warm["GraphName"] = df_warm["Graph"].map(name_map).fillna(df_warm["Graph"])

    # Median per group
    med = (
        df_warm.groupby(["Graph", "GraphName", "LibraryMode"], as_index=False)["Time_ms"]
        .median().rename(columns={"Time_ms": "Median_ms"})
    )
    med["LibraryMode"] = pd.Categorical(med["LibraryMode"], categories=order_modes, ordered=True)
    present = [m for m in order_modes if m in med["LibraryMode"].values]

    # --- Chart 1: all datasets grouped bar
    graphs_ordered = ["LiveJournal", "Patents", "wiki-Talk", "RGG"]
    graphs_present  = [g for g in graphs_ordered if g in med["GraphName"].values]

    fig, ax = plt.subplots(figsize=(13, 6))
    subset = med[med["GraphName"].isin(graphs_present)].sort_values(["GraphName", "LibraryMode"])
    sns.barplot(data=subset, x="GraphName", y="Median_ms",
                hue="LibraryMode", hue_order=present,
                order=graphs_present, palette="Set2", ax=ax)
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(f"{h:.0f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=7)
    ax.set_title("BFS median time — all datasets (runs 2+, warmup dropped)")
    ax.set_ylabel("Median time (ms)")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Mode")
    fig.tight_layout()
    out = BFS_PICS / "bfs_all_datasets.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Plot: {out}")

    # --- Chart 2: box plots (distribution) for LiveJournal
    lj = df_warm[df_warm["Graph"] == "soc-LiveJournal1.mtx"].copy()
    lj["LibraryMode"] = pd.Categorical(lj["LibraryMode"], categories=present, ordered=True)
    if not lj.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=lj, x="LibraryMode", y="Time_ms", palette="Set2", ax=ax)
        ax.set_title("BFS time distribution — LiveJournal (box plot)")
        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("Mode")
        fig.tight_layout()
        out = BFS_PICS / "bfs_lj_boxplot.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Plot: {out}")

    # --- Chart 3: GTEPS for LiveJournal
    edges = {
        "soc-LiveJournal1.mtx": 68_993_773,
        "patents.mtx":          14_970_767,
        "wiki-Talk.mtx":         5_021_410,
        "rgg_n_2_22_s0.mtx":   30_359_198,
    }
    gteps_rows = []
    for (graph, mode), grp in df_warm.groupby(["Graph", "LibraryMode"]):
        med_ms = grp["Time_ms"].median()
        if graph in edges and med_ms > 0:
            gteps_rows.append({
                "GraphName": name_map.get(graph, graph),
                "LibraryMode": mode,
                "GTEPS": edges[graph] / (med_ms * 1e6),
            })
    gteps_df = pd.DataFrame(gteps_rows)
    if not gteps_df.empty:
        gteps_df["LibraryMode"] = pd.Categorical(
            gteps_df["LibraryMode"], categories=present, ordered=True)
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=gteps_df, x="GraphName", y="GTEPS",
                    hue="LibraryMode", hue_order=present,
                    order=graphs_present, palette="viridis", ax=ax)
        for p in ax.patches:
            h = p.get_height()
            if pd.notna(h) and h > 0:
                ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width()/2, h),
                            ha="center", va="bottom", fontsize=7)
        ax.set_title("BFS throughput — GTEPS (all datasets)")
        ax.set_ylabel("Giga Edges per Second")
        ax.legend(title="Mode")
        fig.tight_layout()
        out = BFS_PICS / "bfs_gteps_all.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Plot: {out}")

    # --- Chart 4: directed vs undirected delta (LAGraph only)
    lg_med = med[med["LibraryMode"].isin(["LAGraph directed", "LAGraph undirected"])].copy()
    if len(lg_med) > 0:
        pivot = lg_med.pivot_table(index="GraphName", columns="LibraryMode",
                                   values="Median_ms").reset_index()
        if "LAGraph directed" in pivot.columns and "LAGraph undirected" in pivot.columns:
            pivot["delta_pct"] = (pivot["LAGraph undirected"] /
                                  pivot["LAGraph directed"] - 1.0) * 100
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in pivot["delta_pct"]]
            ax.bar(pivot["GraphName"], pivot["delta_pct"], color=colors)
            ax.axhline(0, color="black", lw=1)
            ax.set_title("LAGraph BFS: undirected vs directed time delta (%)\n(red = undirected slower)")
            ax.set_ylabel("Delta (%)")
            for i, (x, v) in enumerate(zip(pivot["GraphName"], pivot["delta_pct"])):
                ax.annotate(f"{v:+.0f}%", (i, v), ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=9)
            fig.tight_layout()
            out = BFS_PICS / "bfs_directed_vs_undirected_delta.png"
            fig.savefig(out, dpi=150); plt.close(fig)
            print(f"  Plot: {out}")

    print("\n  BFS summary (median, warmup dropped):")
    print(med[["GraphName", "LibraryMode", "Median_ms"]].to_string(index=False))

# ---------------------------------------------------------------------------
# SSSP experiment
# ---------------------------------------------------------------------------

def run_sssp(runs: int):
    print("\n" + "="*60)
    print("SSSP EXPERIMENT")
    print("="*60)

    omp_path = get_omp_prefix()
    compile_runner(SRC_DIR / "sssp_runner.c", LAGRAPH_SSSP, omp_path)

    try:
        spla_sssp = _find_spla("sssp")
        has_spla = True
    except FileNotFoundError as e:
        print(f"  Warning: {e} — skipping SPLA runs")
        has_spla = False

    rows = []

    for graph_name, is_directed in SSSP_DATASETS:
        dataset = DATASETS / graph_name
        if not dataset.exists():
            print(f"  Skip (missing): {graph_name}")
            continue
        print(f"\n  {graph_name}")

        # LAGraph directed
        print("    LAGraph directed ...")
        for i in range(1, runs + 1):
            out = run_cmd([str(LAGRAPH_SSSP), str(dataset), "1", "0", SSSP_DELTA])
            t = parse_lagraph_sssp_ms(out)
            rows.append([graph_name, "LAGraph directed", "directed", i, t])
            print(f"      run {i}: {t:.1f} ms")

        # LAGraph undirected (symmetrised)
        print("    LAGraph undirected ...")
        for i in range(1, runs + 1):
            out = run_cmd([str(LAGRAPH_SSSP), str(dataset), "0", "1", SSSP_DELTA])
            t = parse_lagraph_sssp_ms(out)
            rows.append([graph_name, "LAGraph undirected", "undirected", i, t])
            print(f"      run {i}: {t:.1f} ms")

        if has_spla:
            # SPLA SSSP supports only undirected graphs (push/pull requires AT)
            print("    SpLA GPU (undirected) ...")
            out = run_cmd([
                str(spla_sssp),
                f"--mtxpath={dataset}",
                f"--niters={runs}",
                "--source=0",
                "--run-ref=false", "--run-cpu=false", "--run-gpu=true",
                "--undirected=true",
            ])
            times = parse_spla_gpu_times(out)
            for i, t in enumerate(times[:runs], start=1):
                rows.append([graph_name, "SpLA GPU", "undirected", i, t])
                print(f"      run {i}: {t:.1f} ms")

    # Save CSV
    csv_path = SSSP_RESULTS / "sssp_results_v2.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Graph", "LibraryMode", "Mode", "Iteration", "Time_ms"])
        w.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    _plot_sssp(rows)
    return rows


def _plot_sssp(rows):
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(rows, columns=["Graph", "LibraryMode", "Mode", "Iteration", "Time_ms"])
    df_warm = df[df["Iteration"] > 1].copy()

    order_modes = ["LAGraph directed", "LAGraph undirected", "SpLA GPU"]
    name_map = {
        "soc-LiveJournal1.mtx": "LiveJournal",
        "patents.mtx":          "Patents",
        "wiki-Talk.mtx":        "wiki-Talk",
    }
    df_warm["GraphName"] = df_warm["Graph"].map(name_map).fillna(df_warm["Graph"])

    med = (
        df_warm.groupby(["Graph", "GraphName", "LibraryMode"], as_index=False)["Time_ms"]
        .median().rename(columns={"Time_ms": "Median_ms"})
    )
    present = [m for m in order_modes if m in med["LibraryMode"].values]
    med["LibraryMode"] = pd.Categorical(med["LibraryMode"], categories=present, ordered=True)

    graphs_ordered  = ["LiveJournal", "Patents", "wiki-Talk"]
    graphs_present  = [g for g in graphs_ordered if g in med["GraphName"].values]

    # --- Chart 1: grouped bar, all datasets
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=med, x="GraphName", y="Median_ms",
                hue="LibraryMode", hue_order=present,
                order=graphs_present, palette="Set2", ax=ax)
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(f"{h:.0f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8)
    ax.set_title(f"SSSP median time (delta={SSSP_DELTA}, warmup dropped) — all datasets")
    ax.set_ylabel("Median time (ms)")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Mode")
    fig.tight_layout()
    out = SSSP_PICS / "sssp_all_datasets.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Plot: {out}")

    # --- Chart 2: box plots
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_warm.copy()
    df_plot["GraphName"] = df_plot["Graph"].map(name_map).fillna(df_plot["Graph"])
    df_plot["Label"] = df_plot["GraphName"] + "\n" + df_plot["LibraryMode"]
    sns.boxplot(data=df_plot, x="GraphName", y="Time_ms",
                hue="LibraryMode", hue_order=present, palette="Set2", ax=ax)
    ax.set_title("SSSP time distribution (box plots)")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    ax.legend(title="Mode")
    fig.tight_layout()
    out = SSSP_PICS / "sssp_boxplots.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Plot: {out}")

    # --- Chart 3: GPU speedup (SpLA vs LAGraph directed)
    if "SpLA GPU" in present:
        pivot = med.pivot_table(index="GraphName", columns="LibraryMode",
                                values="Median_ms").reset_index()
        if "LAGraph directed" in pivot.columns and "SpLA GPU" in pivot.columns:
            pivot["Speedup"] = pivot["LAGraph directed"] / pivot["SpLA GPU"]
            pivot = pivot.dropna(subset=["Speedup"])
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(pivot["GraphName"], pivot["Speedup"], color="#2196F3")
            ax.axhline(1.0, color="red", ls="--", lw=1, label="CPU baseline")
            for bar, val in zip(bars, pivot["Speedup"]):
                ax.annotate(f"{val:.1f}×",
                            (bar.get_x() + bar.get_width()/2, val),
                            ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_title("SSSP: SpLA GPU speedup vs LAGraph CPU (directed)")
            ax.set_ylabel("Speedup ×")
            ax.legend()
            fig.tight_layout()
            out = SSSP_PICS / "sssp_gpu_speedup.png"
            fig.savefig(out, dpi=150); plt.close(fig)
            print(f"  Plot: {out}")

    # --- Chart 4: directed vs undirected LAGraph
    lg = med[med["LibraryMode"].isin(["LAGraph directed", "LAGraph undirected"])].copy()
    if len(lg) >= 2:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=lg, x="GraphName", y="Median_ms",
                    hue="LibraryMode",
                    hue_order=["LAGraph directed", "LAGraph undirected"],
                    palette="Set2", ax=ax)
        ax.set_yscale("log")
        ax.set_title("SSSP: LAGraph directed vs undirected (log scale)")
        ax.set_ylabel("Median time (ms, log scale)")
        ax.legend(title="Mode")
        fig.tight_layout()
        out = SSSP_PICS / "sssp_directed_vs_undirected.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Plot: {out}")

    print("\n  SSSP summary (median, warmup dropped):")
    print(med[["GraphName", "LibraryMode", "Median_ms"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",  choices=["bfs", "sssp", "all"], default="all")
    parser.add_argument("--runs",  type=int, default=15,
                        help="Number of timed runs per configuration (default 15)")
    args = parser.parse_args()

    print(f"Runs per config: {args.runs}")
    print(f"Datasets dir:    {DATASETS}")
    print(f"Results dir:     {RESULTS}")

    if args.algo in ("bfs", "all"):
        run_bfs(args.runs)
    if args.algo in ("sssp", "all"):
        run_sssp(args.runs)

    print("\nDone.")


if __name__ == "__main__":
    main()
