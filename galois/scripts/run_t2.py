"""
Run experiments with fixed T=2 OpenMP threads per MPI process.
Saves raw logs and CSV to results/block2_t2/.
"""
import os
import re
import csv
import subprocess
import threading
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
GR_DIR  = ROOT / "galois_gr"
RAW_DIR = ROOT / "results" / "block2_t2" / "raw"
CSV_DIR = ROOT / "results" / "block2_t2" / "csv"

_default_galois = Path(__file__).resolve().parent.parent.parent / "galois-src" / "build"
GALOIS_BUILD = Path(os.environ.get("GALOIS_BUILD", str(_default_galois)))
BIN_DIR = GALOIS_BUILD / "lonestar" / "analytics" / "distributed"

MPIRUN = os.environ.get("MPIRUN", "/opt/homebrew/opt/mpich/bin/mpirun")
RUNS   = 2
T      = 2

RAW_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "algo": "tc",
        "bin":  BIN_DIR / "triangle-counting" / "triangle-counting-dist",
        "graphs": [
            ("soc-LiveJournal1-sym.gr", "livejournal", ""),
            ("rgg_n_2_20_s0.gr",   "rgg",         ""),
            ("roadNet-PA.gr",      "roadnet",      ""),
        ],
        "extra": ["-symmetricGraph"],
        "stat_prefix": "TC",
    },
]

HOST_COUNTS = [1, 2, 4, 6, 8]

STAT_RE = re.compile(
    r"^STAT,\s*\d+,\s*(\w+),\s*Timer_(\d+),\s*HMAX,\s*(\d+)", re.MULTILINE
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


            print(f"Running {tag} (P={n_hosts}, T={T}) ...", end=" ", flush=True)

            cmd = (
                [MPIRUN, "-n", str(n_hosts),
                 str(cfg["bin"]),
                 "-t", str(T),
                 "--partition=oec",
                 f"--runs={RUNS}"]
                + cfg["extra"]
                + ([extra_arg] if extra_arg else [])
                + [str(gr_path)]
            )

            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                lines = []
                done = threading.Event()

                def _read():
                    for line in proc.stdout:
                        lines.append(line)
                        if "Run_UUID" in line:
                            done.set()
                    done.set()

                t = threading.Thread(target=_read, daemon=True)
                t.start()
                done.wait(timeout=300)
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                t.join(timeout=2)
                output = "".join(lines)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue

            log_path.write_text(output)

            matches = STAT_RE.findall(output)
            times = {}
            for region, run_idx, val in matches:
                if region.upper() == cfg["stat_prefix"].upper():
                    times[int(run_idx)] = int(val)

            if not times:
                print(f"NO STATS — check {log_path.name}", flush=True)
                continue

            kept = {i: v for i, v in times.items() if i > 0}
            print(f"times={list(kept.values())} ms", flush=True)

            for run_i, v in kept.items():
                records.append({
                    "algorithm": cfg["algo"],
                    "graph":     gr_name,
                    "num_hosts": n_hosts,
                    "run":       run_i,
                    "time_ms":   v,
                })

csv_path = CSV_DIR / "block2_raw.csv"
fieldnames = ["algorithm", "graph", "num_hosts", "run", "time_ms"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nDone. {len(records)} records → {csv_path}")
