# Block 2: Distributed Graph Processing with Galois

## Goal

Evaluate scalability of distributed graph algorithms by varying the number of compute nodes.
Strong-scaling study: fix the input graph, increase MPI host count (1 → 2 → 4 → 8), measure
wall-clock time, speedup, and parallel efficiency.

---

## Hypothesis

- **Social graphs** (soc-LiveJournal1) will scale poorly beyond 4 hosts due to high edge-cut and
  irregular communication patterns.
- **Geometric graphs** (rgg_n_2_22_s0) will show near-linear speedup because their locality
  maps well to balanced partitions.
- **Road networks** (roadNet-CA) are bottlenecked by high diameter; BFS/SSSP will barely
  benefit from more hosts.
- PageRank is all-reduce-heavy; its scaling depends more on network bandwidth than partition quality.

---

## Datasets

| Graph | Nodes | Edges | Type | Reason chosen |
|---|---|---|---|---|
| soc-LiveJournal1 | 4 847 571 | 68 993 773 | Social | Large, irregular, hard to partition |
| rgg_n_2_22_s0 | 4 194 304 | 30 359 198 | Geometric random | Good locality, expected to scale well |
| roadNet-CA | 1 971 281 | 2 766 607 | Road | High diameter, low degree |

All datasets are in `graph-exp-ws/datasets/` in `.mtx` format. Must be converted to Galois
`.gr` format before running (see Conversion section below).

---

## Algorithms

| Algorithm | Galois binary | Why |
|---|---|---|
| BFS | `bfs-pull-dist` | Latency-sensitive, diameter-dominated |
| SSSP | `sssp-dist` | Same family, weighted edges |
| PageRank | `pagerank-pull-dist` | Iterative, bandwidth-dominated |
| CC (Connected Components) | `cc-dist` | Simple baseline for scaling |

---

## Experimental Design

### Configuration

```
Hosts:         1, 2, 4, 8
Threads/host:  match physical cores on the cluster node (e.g., 16)
Partition:     oec (Outgoing Edge-Cut) — default for directed graphs
               lvc (Logical Vertex-Cut) — try for better load balance on social graphs
Runs per config: 5 (drop first, report median of remaining 4)
```

### Run matrix

For each `(algorithm, graph, num_hosts, partition)`:
- Convert graph once
- Run 5 times, collect Galois `STAT ... Time` output
- Save to `results/block2/raw/<algo>_<graph>_<hosts>h.log`

### Metrics

- **Time (ms)**: median of 4 runs
- **Speedup**: T(1) / T(n)
- **Parallel efficiency**: Speedup / n
- **GTEPS** (for BFS/SSSP): edges / (time × 10⁶)

---

## Galois Setup

### Build (on cluster)

```bash
git clone https://github.com/IntelligentSoftwareSystems/Galois
cd Galois && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGALOIS_ENABLE_DIST=1
make -j$(nproc) bfs-pull-dist sssp-dist pagerank-pull-dist cc-dist graph-convert
```

### Graph Conversion (MTX → Galois .gr)

`mtx2gr` only supports weighted graphs. For pattern (unweighted) MTX files,
convert via edgelist first:

```bash
# Unweighted (pattern MTX) — extract edgelist, then convert
grep -v '^%' soc-LiveJournal1.mtx | tail -n +2 > lj.edges
graph-convert --edgelist2gr --edgeType=void lj.edges soc-LiveJournal1.gr

# Undirected: convert then symmetrize
grep -v '^%' roadNet-CA.mtx | tail -n +2 > road.edges
graph-convert --edgelist2gr --edgeType=void road.edges roadNet-CA.gr
graph-convert --gr2sgr --edgeType=void roadNet-CA.gr roadNet-CA-sym.gr

# Weighted MTX (e.g. rgg with float weights)
graph-convert --mtx2gr --edgeType=float32 rgg_n_2_22_s0.mtx rgg_n_2_22_s0.gr
```

Already converted files are in `graph-exp-ws/galois_gr/`.

### Running (example: BFS, 4 hosts)

```bash
mpirun -n 4 --hostfile hostfile \
  bfs-pull-dist \
  -t 16 \
  --partition=oec \
  --runs=5 \
  soc-LiveJournal1.gr \
  2>&1 | tee results/block2/raw/bfs_livejournal_4h.log
```

### Parsing Galois Output

Galois prints timing lines like:
```
STAT (null) Time 4321 LOOP_END
```

Use the parsing script in `block2_galois.ipynb` or:
```bash
grep "^STAT.*Time" log.log | awk '{print $4}'
```

---

## Results Directory Layout

```
results/block2/
  raw/          ← raw .log files from cluster runs
  csv/          ← parsed CSVs (one per algorithm)
  plots/        ← generated figures
```

---

## Analysis Plan

1. **Scaling plot** — time vs. num_hosts for each (algo, graph) pair
2. **Speedup plot** — speedup vs. num_hosts; include ideal linear reference
3. **Efficiency plot** — parallel efficiency vs. num_hosts
4. **Partition comparison** — oec vs. lvc on LiveJournal (highlight communication overhead)
5. **Algorithm comparison** — fixed graph (LiveJournal), all 4 algorithms at 8 hosts

---

## Notes

- Galois counts partitioned ghost vertices separately; real vertex count stays fixed.
- For PageRank, use `--maxIterations=50 --tolerance=1e-6` for consistency.
- The `--runs=5` flag makes Galois repeat the computation; take lines with `LOOP_END` stat.
- SSSP on unweighted graphs: use unit weights (Galois default for `.gr` without weights).
- rgg graph already has float weights in .mtx; convert with `--convertMtxToGr` (weights preserved).
