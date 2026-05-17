"""Analyse block2 results and generate plots."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import sys

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

ROOT     = Path(__file__).resolve().parent.parent
CSV_DIR  = ROOT / "results" / "block2_t2" / "csv"
PLOT_DIR = ROOT / "results" / "block2_t2" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── load & aggregate ──────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_DIR / "block2_raw.csv")

df = (
    df_raw
    .groupby(["algorithm", "graph", "num_hosts"])["time_ms"]
    .agg(["median", "count"])
    .reset_index()
    .rename(columns={"median": "median_ms",})
)

base = (
    df[df["num_hosts"] == 1][["algorithm", "graph", "median_ms"]]
    .rename(columns={"median_ms": "t1_ms"})
)
df = df.merge(base, on=["algorithm", "graph"])
# time_ratio = T(P)/T(1): < 1 means faster, > 1 means slower
df["time_ratio"] = df["median_ms"] / df["t1_ms"]
# speedup = T(1)/T(P): > 1 means faster (kept for heatmap/efficiency)
df["speedup"]    = df["t1_ms"] / df["median_ms"]
df["efficiency"] = df["speedup"] / df["num_hosts"]

df.to_csv(CSV_DIR / "block2_agg.csv", index=False)
print(df.to_string(index=False))

# ── helpers ───────────────────────────────────────────────────────────────────
GRAPH_LABELS = {"livejournal": "LiveJournal", "rgg": "RGG (2^20)", "roadnet": "RoadNet-PA"}
ALGO_ORDER   = ["tc"]
HOST_COUNTS  = sorted(df["num_hosts"].unique())
COLORS       = sns.color_palette("tab10")

def fmt_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(HOST_COUNTS)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

# ── plot: execution time — one file per algorithm ─────────────────────────────
for algo in algos:
    fig, ax = plt.subplots(figsize=(5, 4))
    sub = df[df["algorithm"] == algo].sort_values("num_hosts")
    for i, (graph_key, grp) in enumerate(sub.groupby("graph")):
        grp = grp.sort_values("num_hosts")
        label = GRAPH_LABELS.get(graph_key, graph_key)
        ax.plot(grp["num_hosts"], grp["median_ms"], marker="o",
                color=COLORS[i], label=label, linewidth=1.5)
    fmt_ax(ax, f"{algo.upper()} — Execution Time", "MPI processes", "Median time (ms)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"time_{algo}.pdf", bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"time_{algo}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: time_{algo}")

# ── plot: time ratio T(P)/T(1) — one file per algorithm ──────────────────────
for algo in algos:
    fig, ax = plt.subplots(figsize=(5, 4))
    # ideal line: T(P)/T(1) = 1/P
    p_arr = np.array(HOST_COUNTS, dtype=float)
    ax.plot(p_arr, 1.0 / p_arr, "k--", linewidth=1, label="Ideal (1/P)", zorder=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="No change")
    sub = df[df["algorithm"] == algo]
    for i, (graph_key, grp) in enumerate(sub.groupby("graph")):
        grp = grp.sort_values("num_hosts")
        label = GRAPH_LABELS.get(graph_key, graph_key)
        ax.plot(grp["num_hosts"], grp["time_ratio"], marker="o",
                color=COLORS[i], label=label, linewidth=1.5)
    fmt_ax(ax, f"{algo.upper()} — Normalized Time T(P)/T(1)",
           "MPI processes", "T(P) / T(1)  [lower = faster]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"speedup_{algo}.pdf", bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"speedup_{algo}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: speedup_{algo}")

# ── plot: efficiency — one file per algorithm ────────────────────────────────
for algo in algos:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal E=1")
    sub = df[df["algorithm"] == algo]
    for i, (graph_key, grp) in enumerate(sub.groupby("graph")):
        grp = grp.sort_values("num_hosts")
        label = GRAPH_LABELS.get(graph_key, graph_key)
        ax.plot(grp["num_hosts"], grp["efficiency"], marker="o",
                color=COLORS[i], label=label, linewidth=1.5)
    fmt_ax(ax, f"{algo.upper()} — Parallel Efficiency E(P) = S(P)/P",
           "MPI processes", "Efficiency")
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"efficiency_{algo}.pdf", bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"efficiency_{algo}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: efficiency_{algo}")


# ── heatmap speedup (traditional T(1)/T(P)) at max processes ─────────────────
df_top = df[df["num_hosts"] == HOST_COUNTS[-1]].copy()
df_top["Graph"] = df_top["graph"].map(GRAPH_LABELS)
df_top["Algo"]  = df_top["algorithm"].str.upper()
pivot = df_top.pivot(index="Algo", columns="Graph", values="speedup").fillna(float("nan"))

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=HOST_COUNTS[-1], ax=ax,
            linewidths=0.5, cbar_kws={"label": "Speedup T(1)/T(P)"})
ax.set_title(f"Speedup T(1)/T(P) at {HOST_COUNTS[-1]} MPI processes (ideal = {HOST_COUNTS[-1]})")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(PLOT_DIR / "speedup_heatmap.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "speedup_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: speedup_heatmap")

# ── BFS throughput ────────────────────────────────────────────────────────────
GRAPH_EDGES = {"livejournal": 68_993_773, "rgg": 7_335_740, "roadnet": 3_083_796}
df_bfs = df[df["algorithm"] == "bfs"].copy()
df_bfs["edges"] = df_bfs["graph"].map(GRAPH_EDGES)
df_bfs["GTEPS"] = df_bfs["edges"] / (df_bfs["median_ms"] * 1e6)

fig, ax = plt.subplots(figsize=(6, 4))
for i, (graph_key, grp) in enumerate(df_bfs.groupby("graph")):
    grp = grp.sort_values("num_hosts")
    ax.plot(grp["num_hosts"], grp["GTEPS"], marker="o",
            color=COLORS[i], label=GRAPH_LABELS.get(graph_key, graph_key),
            linewidth=1.5)
ax.set_title("BFS Throughput (GTEPS)")
ax.set_xlabel("MPI processes")
ax.set_ylabel("GTEPS")
ax.set_xticks(HOST_COUNTS)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "bfs_gteps.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "bfs_gteps.png", bbox_inches="tight")
plt.close()
print("Saved: bfs_gteps")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n=== Time ratio T(P)/T(1) at max scale ===")
pivot_ratio = df_top.pivot(index="Algo", columns="Graph", values="time_ratio").fillna(float("nan"))
print((pivot_ratio).round(2).to_string())
print("\n=== Speedup T(1)/T(P) at max scale ===")
print(pivot.round(2).to_string())
print("\nEfficiency (%):")
eff_pivot = df_top.pivot(index="Algo", columns="Graph", values="efficiency")
print((eff_pivot * 100).round(1).to_string())
