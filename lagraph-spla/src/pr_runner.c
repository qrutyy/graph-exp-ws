#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/*
 * LAGraph PageRank runner with optional per-iteration profiling.
 *
 * Usage:
 *   pr_runner <matrix.mtx> [--niters N] [--profile]
 *
 * --niters N  : run N timed iterations (default 15)
 * --profile   : enable GxB_BURBLE for one separate profiling run;
 *               profiling output goes to stderr so CSV stays clean
 *
 * Output (stdout):
 *   RESULT_SUCCESS
 *   ITERS: <pr_iters>
 *   TOTAL_TIME_MS: <ms>
 *   MS_PER_ITER: <ms>
 *   PER_ITER_MS: <ms0>,<ms1>,...   (only in --profile mode, one line per run)
 */

/* ---------- helpers -------------------------------------------------- */

static double wall_time(void) { return LAGraph_WallClockTime(); }

/* Run LAGr_PageRank once and return wall-clock ms; iters_out = PR iters taken */
static double run_pr(LAGraph_Graph G, int itermax, int *iters_out)
{
    GrB_Vector centrality = NULL;
    int iters_taken = 0;
    char msg[1024];
    double t1 = wall_time();
    int rc = LAGr_PageRank(&centrality, &iters_taken, G, 0.85f, 1e-4f, itermax, msg);
    double t2 = wall_time();
    GrB_free(&centrality);
    if (rc != GrB_SUCCESS) {
        fprintf(stderr, "LAGr_PageRank failed: %s\n", msg);
        return -1.0;
    }
    if (iters_out) *iters_out = iters_taken;
    return (t2 - t1) * 1000.0;
}

/* ---------- main ----------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx> [--niters N] [--profile]\n", argv[0]);
        return 1;
    }

    const char *mtxpath = argv[1];
    int niters   = 15;
    int do_profile = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--niters") == 0 && i+1 < argc) {
            niters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--profile") == 0) {
            do_profile = 1;
        }
    }

    char msg[1024];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    /* ---- load graph ---- */
    GrB_Matrix A = NULL;
    FILE *f = fopen(mtxpath, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", mtxpath); return 1; }
    if (LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) {
        fprintf(stderr, "MMRead failed: %s\n", msg); return 1;
    }
    fclose(f);

    LAGraph_Graph G = NULL;
    LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
    LAGraph_Cached_OutDegree(G, msg);
    LAGraph_Cached_AT(G, msg);  /* needed for push-pull and internal caching */

    /* ---- warmup run (dropped from stats) ---- */
    run_pr(G, 100, NULL);

    /* ---- timed runs ---- */
    double times[64];
    int pr_iters = 0;
    for (int i = 0; i < niters; i++) {
        times[i] = run_pr(G, 100, &pr_iters);
        if (times[i] < 0) return 1;
    }

    /* compute median */
    double sorted[64];
    memcpy(sorted, times, niters * sizeof(double));
    for (int i = 0; i < niters-1; i++)
        for (int j = i+1; j < niters; j++)
            if (sorted[j] < sorted[i]) { double t=sorted[i]; sorted[i]=sorted[j]; sorted[j]=t; }
    double median = (niters % 2)
        ? sorted[niters/2]
        : (sorted[niters/2-1] + sorted[niters/2]) / 2.0;

    printf("RESULT_SUCCESS\n");
    printf("ITERS: %d\n", pr_iters);
    printf("TOTAL_TIME_MS: %f\n", median);
    printf("MS_PER_ITER: %f\n", median / pr_iters);

    /* per-run list for CSV (all runs) */
    printf("ALL_TIMES_MS:");
    for (int i = 0; i < niters; i++) printf("%s%.3f", i?",":"", times[i]);
    printf("\n");

    /* ---- profiling run with GxB_BURBLE ---- */
    if (do_profile) {
        fprintf(stderr, "=== GxB_BURBLE profiling run: %s ===\n", mtxpath);
        GxB_Global_Option_set(GxB_BURBLE, true);
        int prof_iters = 0;
        double prof_total = run_pr(G, 100, &prof_iters);
        GxB_Global_Option_set(GxB_BURBLE, false);
        fprintf(stderr, "=== PROFILE DONE: %.1f ms, %d PR iters ===\n",
                prof_total, prof_iters);
    }

    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
