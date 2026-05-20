"""
Generate three small benchmark graphs and convert to Galois .gr format.
  social:  Barabasi-Albert ~65K vertices, ~650K edges  (power law)
  rgg:     Random Geometric Graph 2^17=131K vertices, ~900K edges
  road:    Grid graph 320x320 = 102K vertices, ~400K edges (road-like, high diameter)
"""
import random, math
from pathlib import Path
import subprocess, sys

CONV   = Path(__file__).resolve().parent.parent.parent / "galois-src/build/tools/graph-convert/graph-convert"
GR_DIR = Path(__file__).resolve().parent.parent / "galois_gr"

def write_and_convert(name, edges, n_vertices=None, symmetric=False):
    txt = GR_DIR / f"{name}.txt"
    gr  = GR_DIR / f"{name}.gr"
    sgr = GR_DIR / f"{name}-sym.gr"

    with open(txt, "w") as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")

    subprocess.run([str(CONV), "--edgelist2gr", str(txt), str(gr)],
                   capture_output=True, check=True)
    txt.unlink()

    if symmetric:
        subprocess.run([str(CONV), "--gr2sgr", str(gr), str(sgr)],
                       capture_output=True, check=True)

    size = gr.stat().st_size // 1024
    print(f"  {name}.gr  {size} KB")


# ── 1. Barabasi-Albert social graph  (O(N·M) flat-list sampler) ──────────────
print("[1/3] Generating social (Barabasi-Albert)...")
random.seed(42)
N_BA = 65_000
M    = 10   # edges per new node → avg degree ≈ 20, ~650K edges

# flat list: each node appears once per edge endpoint → O(1) preferential sample
bag = list(range(M)) * 2   # seed clique approximation
edges_ba = []
for new in range(M, N_BA):
    targets = set()
    while len(targets) < M:
        targets.add(bag[random.randrange(len(bag))])
    for t in targets:
        edges_ba.append((new, t))
        edges_ba.append((t, new))
        bag.append(new)
        bag.append(t)

write_and_convert("social_ba", edges_ba, symmetric=True)
print(f"    {N_BA} vertices, {len(edges_ba)//2} undirected edges")


# ── 2. RGG 2^17 ───────────────────────────────────────────────────────────────
print("[2/3] Generating RGG (n=2^17)...")
random.seed(0)
N_RGG = 1 << 17        # 131 072
R2 = 7.0 / (math.pi * N_RGG)
R  = math.sqrt(R2)

xs = [random.random() for _ in range(N_RGG)]
ys = [random.random() for _ in range(N_RGG)]
grid = {}
for i in range(N_RGG):
    cx, cy = int(xs[i] / R), int(ys[i] / R)
    grid.setdefault((cx, cy), []).append(i)

edges_rgg = []
for i in range(N_RGG):
    cx, cy = int(xs[i] / R), int(ys[i] / R)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for j in grid.get((cx+dx, cy+dy), []):
                if j > i:
                    d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                    if d2 <= R2:
                        edges_rgg.append((i, j))
                        edges_rgg.append((j, i))

write_and_convert("rgg_small", edges_rgg, symmetric=True)
print(f"    {N_RGG} vertices, {len(edges_rgg)//2} undirected edges")


# ── 3. Grid graph (road-like) ─────────────────────────────────────────────────
print("[3/3] Generating grid (road-like)...")
W, H   = 320, 320      # 102 400 vertices, diameter = W+H-2 = 638
edges_grid = []
for r in range(H):
    for c in range(W):
        v = r * W + c
        if c + 1 < W:
            edges_grid.append((v, v + 1))
            edges_grid.append((v + 1, v))
        if r + 1 < H:
            edges_grid.append((v, v + W))
            edges_grid.append((v + W, v))

write_and_convert("grid_small", edges_grid, symmetric=True)
print(f"    {W*H} vertices, {len(edges_grid)//2} undirected edges")

print("\nDone.")
