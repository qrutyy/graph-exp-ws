"""
Run BFS push vs BFS pull comparison with T=2.
Saves raw logs and CSV to results/block2_bfs_push/.
"""
import os
import re
import csv
import subprocess
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
GR_DIR  = ROOT / "galois_gr"
RAW_DIR = ROOT / "results" / "block2_bfs_push" / "raw"
CSV_DIR = ROOT / "results" / "block2_bfs_push" / "csv"

_default_galois = Path(__file__).resolve().parent.parent.parent / "galois-src" / "build"
GALOIS_BUILD = Path(os.environ.get("GALOIS_BUILD", str(_default_galois)))
BIN_DIR = GALOIS_BUILD / "lonestar" / "analytics" / "distributed"

MPIRUN = os.environ.get("MPIRUN", "/opt/homebrew/opt/mpich/bin/mpirun")
RUNS   = 3
T      = 2

RAW_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Test both pull and push on same graphs with same T=2
VARIANTS = [
    {
        "variant": "pull",
        "bin": BIN_DIR / "bfs" / "bfs-pull-dist",
        "stat_prefix": "BFS",
    },
    {
        "variant": "push",
        "bin": BIN_DIR / "bfs" / "bfs-push-dist",
        "stat_prefix": "BFS",
    },
]

GRAPHS = [
    ("soc-LiveJournal1.gr", "livejournal", "--startNode=10"),
    ("roadNet-CA-sym.gr",   "roadnet",     "--startNode=1000"),
    ("rgg_n_2_22_s0.gr",   "rgg",          "--startNode=10"),
]

HOST_COUNTS = [1, 2, 4, 6, 8]

STAT_RE = re.compile(
    r"^STAT,\s*\d+,\s*(\w+)_(\d+),\s*Time,\s*HMAX,\s*(\d+)", re.MULTILINE
)

records = []

for var in VARIANTS:
    if not var["bin"].exists():
        print(f"SKIP (no binary): {var['bin']}", flush=True)
        continue
    for gr_file, gr_name, extra_arg in GRAPHS:
        gr_path = GR_DIR / gr_file
        if not gr_path.exists():
            print(f"  SKIP (missing graph): {gr_file}", flush=True)
            continue

        for n_hosts in HOST_COUNTS:
            tag = f"bfs_{var['variant']}_{gr_name}_{n_hosts}h"
            log_path = RAW_DIR / f"{tag}.log"

            print(f"Running {tag} (P={n_hosts}, T={T}) ...", end=" ", flush=True)

            cmd = (
                [MPIRUN, "-n", str(n_hosts),
                 str(var["bin"]),
                 "-t", str(T),
                 "--partition=oec",
                 f"--runs={RUNS}",
                 extra_arg,
                 str(gr_path)]
            )

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )
                output = result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                print("TIMEOUT", flush=True)
                continue
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue

            log_path.write_text(output)

            matches = STAT_RE.findall(output)
            times = {}
            for region, run_idx, val in matches:
                if region.upper() == var["stat_prefix"].upper():
                    times[int(run_idx)] = int(val)

            if not times:
                print(f"NO STATS — check {log_path.name}", flush=True)
                continue

            kept = {i: v for i, v in times.items() if i > 0}
            print(f"times={list(kept.values())} ms", flush=True)

            for run_i, v in kept.items():
                records.append({
                    "variant":   var["variant"],
                    "graph":     gr_name,
                    "num_hosts": n_hosts,
                    "run":       run_i,
                    "time_ms":   v,
                })

csv_path = CSV_DIR / "bfs_push_vs_pull.csv"
fieldnames = ["variant", "graph", "num_hosts", "run", "time_ms"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nDone. {len(records)} records → {csv_path}")
