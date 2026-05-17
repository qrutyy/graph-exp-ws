"""
Produce a full call-trace breakdown table (compute / sync / rest / total)
for all 45 (algo × graph × host_count) combinations, with percentages.
Prints Markdown and saves results/block2/csv/profile_breakdown.csv.
"""
import re, csv
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "block2" / "raw"
CSV_DIR = ROOT / "results" / "block2" / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

HOST_COUNTS = [1, 2, 4, 6, 8]
GRAPHS      = ["livejournal", "rgg", "roadnet"]
ALGOS       = ["bfs", "pagerank", "cc"]
RUN         = 1   # post-warmup run

GRAPH_LABELS = {"livejournal": "LiveJournal", "rgg": "RGG", "roadnet": "RoadNet-CA"}

# ── parser ────────────────────────────────────────────────────────────────────

def parse_stat(text):
    d = {}
    for line in text.splitlines():
        m = re.match(r"^STAT,\s*\d+,\s*(.+?),\s*(.+?),\s*\w+,\s*([\d.]+)", line)
        if m:
            d[(m.group(1).strip(), m.group(2).strip())] = float(m.group(3))
    return d

def g(d, region, cat, default=0.0):
    return d.get((region, cat), default)

# ── extract one row ───────────────────────────────────────────────────────────

def extract(algo, graph, n):
    path = RAW_DIR / f"{algo}_{graph}_{n}h.log"
    if not path.exists():
        return None
    s = parse_stat(path.read_text())
    r = RUN

    if algo == "bfs":
        sync   = g(s, "Gluon", f"Sync_BFS_{r}")
        reset  = 0.0
        iters  = g(s, "BFS",  f"NumIterations_{r}")
        workM  = g(s, "BFS",  f"NumWorkItems_{r}") / 1e6
        total  = g(s, "BFS",  f"Timer_{r}")
    elif algo == "pagerank":
        sync   = g(s, "Gluon", f"Sync_PageRank_{r}")
        reset  = g(s, f"RESET:MIRRORS_{r}", "Time")
        iters  = g(s, "PageRank", f"NumIterations_{r}")
        workM  = g(s, "PageRank", f"NumWorkItems_{r}") / 1e6
        total  = g(s, "PageRank", f"Timer_{r}")
    elif algo == "cc":
        sync   = g(s, "Gluon", f"Sync_ConnectedComp_{r}")
        reset  = 0.0
        iters  = g(s, "ConnectedComp", f"NumIterations_{r}")
        workM  = g(s, "ConnectedComp", f"NumWorkItems_{r}") / 1e6
        total  = g(s, "ConnectedComp", f"Timer_{r}")
    else:
        return None

    # compute = everything that isn't sync or reset overhead
    compute = max(0.0, total - sync - reset)
    denom   = total if total > 0 else 1.0

    return dict(
        algo=algo, graph=graph, n_hosts=n,
        compute=compute, sync=sync, reset=reset, total=total,
        pct_compute=compute/denom*100, pct_sync=sync/denom*100,
        pct_reset=reset/denom*100,
        iters=int(iters), work_M=workM,
    )

# ── collect all rows ──────────────────────────────────────────────────────────

rows = []
for algo in ALGOS:
    for graph in GRAPHS:
        for n in HOST_COUNTS:
            row = extract(algo, graph, n)
            if row:
                rows.append(row)

# ── save CSV ──────────────────────────────────────────────────────────────────

fields = ["algo","graph","n_hosts","compute","sync","reset","total",
          "pct_compute","pct_sync","pct_reset","iters","work_M"]
with open(CSV_DIR / "profile_breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── print Markdown tables (one per algo) ─────────────────────────────────────

for algo in ALGOS:
    has_reset = (algo == "pagerank")
    if has_reset:
        header = ("| Граф | Proc | Compute мс | Compute% | "
                  "Reset мс | Reset% | Sync мс | **Sync%** | **Итого мс** | Итер. | Work (M) |")
        sep    =  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = ("| Граф | Proc | Compute мс | Compute% | "
                  "Sync мс | **Sync%** | **Итого мс** | Итер. | Work (M) |")
        sep    =  "|---|---|---:|---:|---:|---:|---:|---:|---:|"

    print(f"\n### {algo.upper()}\n")
    print(header)
    print(sep)
    prev_graph = None
    for row in rows:
        if row["algo"] != algo:
            continue
        gl  = GRAPH_LABELS[row["graph"]]
        n   = row["n_hosts"]
        c   = row["compute"];  pc = row["pct_compute"]
        sy  = row["sync"];     ps = row["pct_sync"]
        re_ = row["reset"];    pr = row["pct_reset"]
        tot = row["total"]
        it  = row["iters"]
        wm  = row["work_M"]
        # blank separator row between graphs
        if prev_graph and prev_graph != row["graph"]:
            print(sep.replace("---:", "   ").replace("|---|", "|   |"))
        prev_graph = row["graph"]
        sync_fmt = f"**{sy:.0f}**" if ps >= 50 else f"{sy:.0f}"
        pct_fmt  = f"**{ps:.0f}%**" if ps >= 50 else f"{ps:.0f}%"
        if has_reset:
            print(f"| {gl} | {n} | {c:.0f} | {pc:.0f}% | "
                  f"{re_:.0f} | {pr:.0f}% | {sync_fmt} | {pct_fmt} | "
                  f"**{tot:.0f}** | {it} | {wm:.1f} |")
        else:
            print(f"| {gl} | {n} | {c:.0f} | {pc:.0f}% | "
                  f"{sync_fmt} | {pct_fmt} | **{tot:.0f}** | {it} | {wm:.1f} |")
