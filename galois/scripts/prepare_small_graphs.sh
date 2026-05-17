#!/usr/bin/env bash
# Download and convert small datasets (~10x smaller than original)
# Social: web-BerkStan  (685K vertices, 7.6M edges  — power law, replaces LJ)
# Geo:    rgg_n_2_20    (1M vertices,   ~7M edges    — geometric, replaces rgg_22)
# Road:   roadNet-PA    (1.1M vertices, 1.5M edges   — road, replaces roadNet-CA)
set -e

GR_DIR="$(cd "$(dirname "$0")/../galois_gr" && pwd)"
CONV=/Users/qruty/prog/graphs/galois-src/build/tools/graph-convert/graph-convert
TMP=$(mktemp -d)

echo "Output dir: $GR_DIR"
echo "Temp dir:   $TMP"

# ── 1. web-BerkStan (social/web, power law) ───────────────────────────────────
if [ ! -f "$GR_DIR/web-BerkStan.gr" ]; then
  echo "[1/3] Downloading web-BerkStan..."
  curl -L -o "$TMP/web-BerkStan.txt.gz" \
    "https://snap.stanford.edu/data/web-BerkStan.txt.gz"
  gunzip "$TMP/web-BerkStan.txt.gz"
  echo "  Converting to .gr..."
  $CONV --edgelist2gr "$TMP/web-BerkStan.txt" "$GR_DIR/web-BerkStan.gr"
  echo "  Creating symmetric version..."
  $CONV --gr2sgr "$GR_DIR/web-BerkStan.gr" "$GR_DIR/web-BerkStan-sym.gr"
  echo "  Done: web-BerkStan"
else
  echo "[1/3] web-BerkStan already exists, skipping"
fi

# ── 2. roadNet-PA (road network, ~1.1M vertices) ──────────────────────────────
if [ ! -f "$GR_DIR/roadNet-PA.gr" ]; then
  echo "[2/3] Downloading roadNet-PA..."
  curl -L -o "$TMP/roadNet-PA.txt.gz" \
    "https://snap.stanford.edu/data/roadNet-PA.txt.gz"
  gunzip "$TMP/roadNet-PA.txt.gz"
  echo "  Converting to .gr..."
  $CONV --edgelist2gr "$TMP/roadNet-PA.txt" "$GR_DIR/roadNet-PA.gr"
  echo "  Creating symmetric version..."
  $CONV --gr2sgr "$GR_DIR/roadNet-PA.gr" "$GR_DIR/roadNet-PA-sym.gr"
  echo "  Done: roadNet-PA"
else
  echo "[2/3] roadNet-PA already exists, skipping"
fi

# ── 3. RGG n=2^20 ≈ 1M vertices (geometric, replaces rgg_n_2_22) ─────────────
if [ ! -f "$GR_DIR/rgg_n_2_20_s0.gr" ]; then
  echo "[3/3] Generating rgg_n_2_20_s0..."
  python3 - <<'PYEOF'
import random, math, sys
from pathlib import Path

random.seed(0)
N = 1 << 20          # 2^20 = 1 048 576 vertices
# same avg degree ~7 as rgg_22: r = sqrt(7 / (pi * N))
R2 = 7.0 / (math.pi * N)
R  = math.sqrt(R2)

# Use a grid to find neighbors in O(N) expected time
cell = int(1.0 / R) + 1
grid = {}
xs = [random.random() for _ in range(N)]
ys = [random.random() for _ in range(N)]
for i, (x, y) in enumerate(zip(xs, ys)):
    cx, cy = int(x / R), int(y / R)
    grid.setdefault((cx, cy), []).append(i)

edges = []
for i in range(N):
    cx = int(xs[i] / R)
    cy = int(ys[i] / R)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for j in grid.get((cx+dx, cy+dy), []):
                if j != i:
                    d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                    if d2 <= R2:
                        edges.append((i, j))

out = Path(__file__).parent.parent / "galois_gr" / "rgg_n_2_20_s0.txt"
with open(out, "w") as f:
    for u, v in edges:
        f.write(f"{u}\t{v}\n")
print(f"  Generated {N} vertices, {len(edges)} directed edges -> {out}")
PYEOF
  echo "  Converting to .gr..."
  $CONV --edgelist2gr "$GR_DIR/rgg_n_2_20_s0.txt" "$GR_DIR/rgg_n_2_20_s0.gr"
  rm "$GR_DIR/rgg_n_2_20_s0.txt"
  echo "  Done: rgg_n_2_20_s0"
else
  echo "[3/3] rgg_n_2_20_s0 already exists, skipping"
fi

rm -rf "$TMP"
echo ""
echo "All done. Files in $GR_DIR:"
ls -lh "$GR_DIR/"*.gr | grep -E "BerkStan|PA|rgg.*20"
