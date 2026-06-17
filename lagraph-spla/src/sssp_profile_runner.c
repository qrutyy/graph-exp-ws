#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/*
 * SSSP per-round profiling runner (delta-stepping, delta=1.0, unit weights).
 *
 * With delta=1.0 and unit weights, every edge is "light" — delta-stepping
 * degenerates to BFS-style processing: round k handles vertices with
 * tentative distance d in [k, k+1). This gives the same level structure
 * as BFS for unweighted graphs.
 *
 * Usage: sssp_profile_runner <matrix.mtx> [source]
 * Output CSV: round,bucket_size,time_ms,settled_total
 */

static GrB_Index find_max_out_degree_node(LAGraph_Graph G)
{
    GrB_Index n = 0;
    GrB_Matrix_nrows(&n, G->A);
    GrB_Index *nids = (GrB_Index *)malloc(n * sizeof(GrB_Index));
    int64_t   *degs = (int64_t  *)malloc(n * sizeof(int64_t));
    GrB_Index  nv   = n;
    GrB_Vector_extractTuples_INT64(nids, degs, &nv, G->out_degree);
    GrB_Index best = 0; int64_t bd = -1;
    for (GrB_Index i = 0; i < nv; i++)
        if (degs[i] > bd) { bd = degs[i]; best = nids[i]; }
    free(nids); free(degs);
    return best;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <matrix.mtx> [src]\n", argv[0]); return 1; }

    char msg[LAGRAPH_MSG_LEN];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    /* load and convert to FP32 unit weights */
    GrB_Matrix A_raw = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A_raw, f, msg) != GrB_SUCCESS) {
        fprintf(stderr, "Load failed: %s\n", msg); return 1;
    }
    fclose(f);

    GrB_Index n = 0;
    GrB_Matrix_nrows(&n, A_raw);

    GrB_Matrix A = NULL;
    GrB_Matrix_new(&A, GrB_FP32, n, n);
    GrB_assign(A, A_raw, NULL, (float)1.0f, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S);
    GrB_free(&A_raw);

    /* also build symmetrised version for undirected semantics */
    GrB_Matrix AT = NULL;
    GrB_Matrix_new(&AT, GrB_FP32, n, n);
    GrB_transpose(AT, NULL, NULL, A, NULL);
    GrB_eWiseAdd(A, NULL, NULL, GrB_MIN_FP32, A, AT, NULL);
    GrB_free(&AT);
    GrB_wait(A, GrB_MATERIALIZE);

    LAGraph_Graph G = NULL;
    LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
    LAGraph_Cached_OutDegree(G, msg);

    GrB_Index source = (argc > 2) ? (GrB_Index)atol(argv[2])
                                   : find_max_out_degree_node(G);
    fprintf(stderr, "source=%llu\n", (unsigned long long)source);

    /* ---- distance vector: inf everywhere, 0 at source ---- */
    GrB_Vector dist = NULL;
    GrB_Vector_new(&dist, GrB_FP32, n);
    GrB_assign(dist, NULL, NULL, (float)FLT_MAX, GrB_ALL, n, NULL);
    GrB_Vector_setElement_FP32(dist, 0.0f, source);
    GrB_wait(dist, GrB_MATERIALIZE);

    /* ---- simplified delta-stepping with delta=1.0 ---- *
     * Since all weights=1 and delta=1, every edge is light.
     * Round k: active = vertices with dist in [k, k+1)
     * Relax out-edges; settle when no improvement possible.
     */
    printf("round,bucket_size,time_ms,settled_total\n");

    float delta = 1.0f;
    GrB_Index settled = 1;  /* source is settled */
    int round = 0;
    int max_rounds = 5000;
    int empty_streak = 0;   /* consecutive empty buckets → termination */

    /* Descriptor for masked assign/select */
    GrB_Descriptor desc_str = NULL;
    GrB_Descriptor_new(&desc_str);
    GrB_Descriptor_set(desc_str, GrB_MASK, GrB_STRUCTURE);

    while (round < max_rounds) {
        float lo = round * delta;
        float hi = lo + delta;

        /* extract bucket: vertices with dist in [lo, hi) */
        GrB_Vector bucket = NULL;
        GrB_Vector_new(&bucket, GrB_FP32, n);
        GrB_select(bucket, NULL, NULL, GrB_VALUEGE_FP32, dist, lo, NULL);
        GrB_select(bucket, NULL, NULL, GrB_VALUELT_FP32, bucket, hi, desc_str);
        GrB_wait(bucket, GrB_MATERIALIZE);

        GrB_Index bsize = 0;
        GrB_Vector_nvals(&bsize, bucket);

        if (bsize == 0) {
            GrB_free(&bucket);
            empty_streak++;
            /* stop after 3 consecutive empty buckets: all reachable vertices settled */
            if (empty_streak >= 3) break;
            round++;
            continue;
        }
        empty_streak = 0;
        round++;
        double t1 = LAGraph_WallClockTime();

        /* relax: d_new[j] = min(d[j], bucket_dist[i] + 1) for edges (i,j) */
        GrB_Vector update = NULL;
        GrB_Vector_new(&update, GrB_FP32, n);
        GrB_mxv(update, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP32, G->A, bucket, NULL);
        GrB_wait(update, GrB_MATERIALIZE);

        /* apply updates: dist = min(dist, update) */
        GrB_eWiseAdd(dist, NULL, GrB_MIN_FP32, GrB_MIN_FP32, dist, update, NULL);
        GrB_wait(dist, GrB_MATERIALIZE);

        double t2 = LAGraph_WallClockTime();

        settled += bsize;
        printf("%d,%llu,%.4f,%llu\n",
               round, (unsigned long long)bsize,
               (t2 - t1) * 1000.0,
               (unsigned long long)settled);
        fflush(stdout);

        GrB_free(&bucket); GrB_free(&update);
    }

    fprintf(stderr, "rounds=%d  settled=%llu / %llu\n",
            round, (unsigned long long)settled, (unsigned long long)n);

    GrB_free(&desc_str);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
