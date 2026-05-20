"""Extract timing breakdowns from raw Galois logs and generate profiling plots."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "results" / "block2" / "raw"
PLOT_DIR = ROOT / "results" / "block2" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HOST_COUNTS = [1, 2, 4, 6, 8]

# ── log parser ────────────────────────────────────────────────────────────────

def parse_stat(log_text):
    """Return dict of {(region, category): value} from STAT lines."""
    result = {}
    for line in log_text.splitlines():
        m = re.match(r"^STAT,\s*\d+,\s*(.+?),\s*(.+?),\s*\w+,\s*([\d.]+)", line)
        if m:
            key = (m.group(1).strip(), m.group(2).strip())
            try:
                result[key] = float(m.group(3))
            except ValueError:
                pass
    return result

def get(d, region, category, default=0.0):
    return d.get((region, category), default)

# ── collect per-run breakdown ─────────────────────────────────────────────────
# Use run index 1 (first post-warmup run) for consistency.

def load_breakdown(algo, graph, run=1):
    rows = {}
    for n in HOST_COUNTS:
        path = RAW_DIR / f"{algo}_{graph}_{n}h.log"
        if not path.exists():
            continue
        s = parse_stat(path.read_text())

        if algo == "pagerank":
            compute = (get(s, f"PageRank_delta_{run}", "Time") +
                       get(s, f"PageRank_{run}", "Time"))
            sync    = get(s, f"Gluon", f"Sync_PageRank_{run}")
            reset   = get(s, f"RESET:MIRRORS_{run}", "Time")
            iters   = get(s, "PageRank", f"NumIterations_{run}")
            nbytes  = get(s, "Gluon", f"ReduceSendBytes_PageRank_{run}")
            total   = get(s, "PageRank", f"Timer_{run}")
        elif algo == "bfs":
            compute = get(s, f"BFS_{run}", "Time")
            sync    = get(s, "Gluon", f"Sync_BFS_{run}")
            reset   = 0.0
            iters   = get(s, "BFS", f"NumIterations_{run}")
            nbytes  = get(s, "Gluon", f"ReduceSendBytes_BFS_{run}")
            total   = get(s, "BFS", f"Timer_{run}")
        elif algo == "cc":
            compute = get(s, f"ConnectedComp_{run}", "Time")
            sync    = get(s, "Gluon", f"Sync_ConnectedComp_{run}")
            reset   = 0.0
            iters   = get(s, "ConnectedComp", f"NumIterations_{run}")
            nbytes  = get(s, "Gluon", f"ReduceSendBytes_ConnectedComp_{run}")
            total   = get(s, "ConnectedComp", f"Timer_{run}")
        else:
            continue

        rows[n] = dict(compute=compute, sync=sync, reset=reset,
                       iters=iters, nbytes=nbytes, total=total)
    return rows


# ── Plot 1: stacked bar — compute / sync / reset for PageRank LiveJournal ─────

pr_lj = load_breakdown("pagerank", "livejournal")

fig, ax = plt.subplots(figsize=(7, 4))
hosts = sorted(pr_lj)
comp  = [pr_lj[n]["compute"] for n in hosts]
sync  = [pr_lj[n]["sync"]    for n in hosts]
reset = [pr_lj[n]["reset"]   for n in hosts]

x = np.arange(len(hosts))
w = 0.55
p1 = ax.bar(x, comp,  w, label="Вычисления (compute)", color="#4e79a7")
p2 = ax.bar(x, sync,  w, bottom=comp,  label="Синхронизация (Gluon sync)", color="#f28e2b")
p3 = ax.bar(x, reset, w, bottom=[c+s for c,s in zip(comp,sync)],
            label="Сброс зеркал (reset mirrors)", color="#e15759")

ax.set_xticks(x)
ax.set_xticklabels([f"{n} proc" for n in hosts])
ax.set_ylabel("Время (мс)")
ax.set_title("PageRank / LiveJournal: разбивка времени по фазам\n(run 1, после прогрева)")
ax.legend(fontsize=9)

# annotate sync %
for i, n in enumerate(hosts):
    tot = comp[i] + sync[i] + reset[i]
    if tot > 0:
        pct = sync[i] / tot * 100
        ax.text(i, tot + 100, f"sync\n{pct:.0f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR / "profile_pagerank_breakdown.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "profile_pagerank_breakdown.png", bbox_inches="tight")
plt.close()
print("Saved: profile_pagerank_breakdown")


# ── Plot 2: sync fraction for BFS and PageRank on LiveJournal ─────────────────

bfs_lj = load_breakdown("bfs", "livejournal")

fig, ax = plt.subplots(figsize=(6, 4))
for data, label, color in [
    (pr_lj, "PageRank", "#f28e2b"),
    (bfs_lj, "BFS",     "#4e79a7"),
]:
    h_list = sorted(data)
    frac = []
    for n in h_list:
        tot = data[n]["compute"] + data[n]["sync"] + data[n]["reset"]
        frac.append(data[n]["sync"] / tot * 100 if tot > 0 else 0)
    ax.plot(h_list, frac, marker="o", linewidth=1.8, label=label, color=color)

ax.set_xlabel("MPI процессы")
ax.set_ylabel("Доля времени синхронизации (%)")
ax.set_title("Доля времени на синхронизацию Gluon\n(LiveJournal)")
ax.set_xticks(HOST_COUNTS)
ax.set_ylim(0, 100)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "profile_sync_fraction.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "profile_sync_fraction.png", bbox_inches="tight")
plt.close()
print("Saved: profile_sync_fraction")


# ── Plot 3: iterations to convergence ─────────────────────────────────────────

cc_lj = load_breakdown("cc", "livejournal")

fig, ax = plt.subplots(figsize=(6, 4))
for data, label, color in [
    (pr_lj, "PageRank / LJ", "#f28e2b"),
    (bfs_lj, "BFS / LJ",     "#4e79a7"),
    (cc_lj,  "CC / LJ",      "#59a14f"),
]:
    h_list = sorted(data)
    iters = [data[n]["iters"] for n in h_list]
    ax.plot(h_list, iters, marker="o", linewidth=1.8, label=label, color=color)

ax.set_xlabel("MPI процессы")
ax.set_ylabel("Итераций до сходимости")
ax.set_title("Число итераций vs. число MPI-процессов\n(граф LiveJournal)")
ax.set_xticks(HOST_COUNTS)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "profile_iterations.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "profile_iterations.png", bbox_inches="tight")
plt.close()
print("Saved: profile_iterations")


# ── Plot 4: network bytes for PageRank LiveJournal ────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
h_list = sorted(pr_lj)
gb = [pr_lj[n]["nbytes"] / 1e9 for n in h_list]
ax.bar(range(len(h_list)), gb, color="#76b7b2")
ax.set_xticks(range(len(h_list)))
ax.set_xticklabels([f"{n} proc" for n in h_list])
ax.set_ylabel("Отправлено данных (ГБ)")
ax.set_title("Объём сетевого трафика (Gluon Reduce)\nPageRank / LiveJournal, run 1")
for i, v in enumerate(gb):
    ax.text(i, v + 0.02, f"{v:.2f} ГБ", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(PLOT_DIR / "profile_network_bytes.pdf", bbox_inches="tight")
plt.savefig(PLOT_DIR / "profile_network_bytes.png", bbox_inches="tight")
plt.close()
print("Saved: profile_network_bytes")


# ── Print summary table ───────────────────────────────────────────────────────
print()
print("=== PageRank / LiveJournal profiling summary (run 1) ===")
print(f"{'Hosts':>6} {'Compute':>10} {'Sync':>10} {'Reset':>8} {'Total':>10} "
      f"{'Sync%':>7} {'Iters':>7} {'NetGB':>8}")
for n in sorted(pr_lj):
    r = pr_lj[n]
    tot = r["compute"] + r["sync"] + r["reset"]
    sp  = r["sync"] / tot * 100 if tot > 0 else 0
    print(f"{n:>6} {r['compute']:>10.0f} {r['sync']:>10.0f} {r['reset']:>8.0f} "
          f"{tot:>10.0f} {sp:>6.1f}% {r['iters']:>7.0f} {r['nbytes']/1e9:>8.2f}")

print()
print("=== BFS / LiveJournal profiling summary (run 1) ===")
print(f"{'Hosts':>6} {'Compute':>10} {'Sync':>10} {'Total':>10} {'Sync%':>7} {'Iters':>7} {'NetMB':>8}")
for n in sorted(bfs_lj):
    r = bfs_lj[n]
    tot = r["compute"] + r["sync"]
    sp  = r["sync"] / tot * 100 if tot > 0 else 0
    print(f"{n:>6} {r['compute']:>10.0f} {r['sync']:>10.0f} {tot:>10.0f} "
          f"{sp:>6.1f}% {r['iters']:>7.0f} {r['nbytes']/1e6:>8.1f}")
