"""
Run Galois distributed graph experiments locally with multiple MPI processes.
Saves raw logs and parsed CSV to results/block2/.
"""
import os
import re
import subprocess
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent          # graph-exp-ws/
GR_DIR  = ROOT / "galois_gr"
RAW_DIR = ROOT / "results" / "block2" / "raw"
CSV_DIR = ROOT / "results" / "block2" / "csv"

# Galois build: set GALOIS_BUILD env var or edit this fallback
_default_galois = Path(__file__).resolve().parent.parent.parent / "galois-src" / "build"
GALOIS_BUILD = Path(os.environ.get("GALOIS_BUILD", str(_default_galois)))
BIN_DIR = GALOIS_BUILD / "lonestar" / "analytics" / "distributed"

# mpirun: set MPIRUN env var or use default Homebrew MPICH path
MPIRUN = os.environ.get("MPIRUN", "/opt/homebrew/opt/mpich/bin/mpirun")
RUNS   = 3

# Constant total parallelism: P * T = NUM_CORES.
# Each MPI process gets floor(NUM_CORES / P) OpenMP threads, so the total
# thread budget stays fixed regardless of how many processes we launch.
# This ensures T(1) is measured at full machine utilisation and S(P) reflects
# pure MPI distribution overhead, not the benefit of switching from 4→12 threads.
NUM_CORES = int(os.environ.get("NUM_CORES", 12))
THREADS_MAP = {n: max(1, NUM_CORES // n) for n in [1, 2, 4, 6, 8]}

RAW_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

CONFIGS = [
    {
        "algo": "bfs",
        "bin":  BIN_DIR / "bfs" / "bfs-pull-dist",
        "graphs": [
            ("soc-LiveJournal1.gr", "livejournal", "--startNode=10"),
            ("rgg_n_2_22_s0.gr",   "rgg",          "--startNode=10"),
            ("roadNet-CA-sym.gr",  "roadnet",       "--startNode=1000"),
        ],
        "extra": [],
        "stat_prefix": "BFS",
    },
    {
        "algo": "pagerank",
        "bin":  BIN_DIR / "pagerank" / "pagerank-pull-dist",
        "graphs": [
            ("soc-LiveJournal1.gr", "livejournal", ""),
            ("rgg_n_2_22_s0.gr",   "rgg",          ""),
            ("roadNet-CA.gr",      "roadnet",       ""),   # directed graph for PageRank
        ],
        "extra": ["--maxIterations=20", "--tolerance=1e-4"],
        "stat_prefix": "PageRank",
    },
    {
        "algo": "cc",
        "bin":  BIN_DIR / "connected-components" / "connected-components-pull-dist",
        "graphs": [
            ("soc-LiveJournal1-sym.gr", "livejournal", "-symmetricGraph"),  # symmetrized
            ("rgg_n_2_22_s0.gr",        "rgg",         "-symmetricGraph"),
            ("roadNet-CA-sym.gr",        "roadnet",     "-symmetricGraph"),
        ],
        "extra": [],
        "stat_prefix": "ConnectedComp",
    },
]

HOST_COUNTS = [1, 2, 4, 6, 8]

# Regex: STAT, HOST_ID, ALGO_N, Time, HMAX, VALUE
STAT_RE = re.compile(
    r"^STAT,\s*\d+,\s*(\w+)_(\d+),\s*Time,\s*HMAX,\s*(\d+)", re.MULTILINE
)

records = []

for cfg in CONFIGS:
    for gr_file, gr_name, extra_arg in cfg["graphs"]:
        gr_path = GR_DIR / gr_file
        if not gr_path.exists():
            print(f"  SKIP (missing): {gr_file}", flush=True)
            continue

        for n_hosts in HOST_COUNTS:
            tag = f"{cfg['algo']}_{gr_name}_{n_hosts}h"
            log_path = RAW_DIR / f"{tag}.log"

            t = THREADS_MAP[n_hosts]
            print(f"Running {tag} (P={n_hosts}, T={t}) ...", end=" ", flush=True)

            cmd = (
                [MPIRUN, "-n", str(n_hosts),
                 str(cfg["bin"]),
                 "-t", str(t),
                 "--partition=oec",
                 f"--runs={RUNS}"]
                + cfg["extra"]
                + ([extra_arg] if extra_arg else [])
                + [str(gr_path)]
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

            # Parse timings: ALGO_0 ... ALGO_{RUNS-1}, skip run 0 (warmup)
            matches = STAT_RE.findall(output)
            times = {}
            for region, run_idx, val in matches:
                if region.upper() == cfg["stat_prefix"].upper():
                    times[int(run_idx)] = int(val)

            if not times:
                print(f"NO STATS — check {log_path.name}", flush=True)
                continue

            kept = {i: t for i, t in times.items() if i > 0}  # drop warmup
            print(f"times={list(kept.values())} ms", flush=True)

            for run_i, t in kept.items():
                records.append({
                    "algorithm": cfg["algo"],
                    "graph":     gr_name,
                    "num_hosts": n_hosts,
                    "run":       run_i,
                    "time_ms":   t,
                })

import csv
csv_path = CSV_DIR / "block2_raw.csv"
fieldnames = ["algorithm", "graph", "num_hosts", "run", "time_ms"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nDone. {len(records)} records → {csv_path}")
