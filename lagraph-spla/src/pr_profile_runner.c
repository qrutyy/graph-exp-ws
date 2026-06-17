#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/*
 * PageRank profiling runner: manual loop timing each phase per iteration.
 *
 * Usage: pr_profile_runner <matrix.mtx>
 *
 * Implements the same algorithm as LAGr_PageRank (alpha=0.85, eps=1e-4).
 * Times each phase: SpMV, eWiseMult, assign/reduce/apply (grouped as "other").
 *
 * Output: CSV rows to stdout:
 *   iter, spMV_ms, scale_ms, other_ms, total_iter_ms, norm_delta
 */

#define ALPHA      0.85f
#define EPS        1e-4f
#define ITERMAX    100

static double wt(void) { return LAGraph_WallClockTime(); }

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    char msg[LAGRAPH_MSG_LEN];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    /* ---- load ---- */
    GrB_Matrix A = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) {
        fprintf(stderr, "Load failed: %s\n", msg); return 1;
    }
    fclose(f);

    GrB_Index n = 0;
    GrB_Matrix_nrows(&n, A);

    /* cast to FP64 for the same semiring LAGraph uses */
    GrB_Matrix AT = NULL;
    GrB_Matrix_new(&AT, GrB_FP64, n, n);
    GrB_transpose(AT, NULL, NULL, A, NULL);
    GrB_free(&A);

    /* ---- compute out-degree and d = alpha / out_degree ---- */
    GrB_Vector d = NULL;   /* per-node scaling: alpha / out_degree */
    GrB_Vector_new(&d, GrB_FP64, n);
    GrB_Matrix_reduce_Monoid(d, NULL, NULL, GrB_PLUS_MONOID_FP64, AT, GrB_DESC_T0);
    /* d[i] = out_degree[i]; replace with alpha/out_degree */
    GrB_apply(d, NULL, NULL, GrB_MINV_FP64, d, NULL);
    GrB_apply(d, NULL, NULL, GrB_TIMES_FP64, d, (double)ALPHA, NULL);

    double teleport = (1.0 - ALPHA) / (double)n;

    /* ---- initialise rank vector r = 1/n ---- */
    GrB_Vector r = NULL, r_new = NULL, t = NULL;
    GrB_Vector_new(&r,     GrB_FP64, n);
    GrB_Vector_new(&r_new, GrB_FP64, n);
    GrB_Vector_new(&t,     GrB_FP64, n);
    GrB_Vector_assign_FP64(r, NULL, NULL, 1.0 / (double)n, GrB_ALL, n, NULL);
    GrB_wait(r, GrB_MATERIALIZE);

    /* ---- print CSV header ---- */
    printf("iter,spMV_ms,eWiseMult_ms,other_ms,total_iter_ms,norm_delta\n");

    int converged = 0;
    for (int iter = 1; iter <= ITERMAX && !converged; iter++) {
        double t0, t1, t2, t3, t4;
        double spMV_ms, ewm_ms, other_ms;

        t0 = wt();

        /* Phase 1: scale r by degree weight: t = r .* d  (O(n)) */
        GrB_eWiseMult(t, NULL, NULL, GrB_TIMES_FP64, r, d, NULL);
        GrB_wait(t, GrB_MATERIALIZE);
        t1 = wt();

        /* Phase 2: SpMV: r_new = AT * t  (O(nnz)) — dominant cost */
        GrB_mxv(r_new, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, AT, t, NULL);
        GrB_wait(r_new, GrB_MATERIALIZE);
        t2 = wt();

        /* Phase 3: add teleportation: r_new[i] += teleport  (O(n)) */
        GrB_assign(r_new, NULL, GrB_PLUS_FP64, teleport, GrB_ALL, n, NULL);
        GrB_wait(r_new, GrB_MATERIALIZE);
        t3 = wt();

        /* Phase 4: convergence check: delta = ||r_new - r||_1  (O(n)) */
        double delta = 0.0;
        GrB_eWiseAdd(t, NULL, NULL, GrB_MINUS_FP64, r_new, r, NULL);
        GrB_apply(t, NULL, NULL, GrB_ABS_FP64, t, NULL);
        GrB_reduce(&delta, NULL, GrB_PLUS_MONOID_FP64, t, NULL);
        GrB_Vector_dup(&r, r_new);   /* r <- r_new */
        GrB_wait(r, GrB_MATERIALIZE);
        t4 = wt();

        ewm_ms    = (t1 - t0) * 1000.0;
        spMV_ms   = (t2 - t1) * 1000.0;
        other_ms  = (t4 - t2) * 1000.0;
        double total_ms = (t4 - t0) * 1000.0;

        printf("%d,%.4f,%.4f,%.4f,%.4f,%.2e\n",
               iter, spMV_ms, ewm_ms, other_ms, total_ms, delta);
        fflush(stdout);

        if (delta < EPS) { converged = 1; }
    }

    fprintf(stderr, "Converged=%d\n", converged);

    GrB_free(&r); GrB_free(&r_new); GrB_free(&t);
    GrB_free(&d); GrB_free(&AT);
    LAGraph_Finalize(msg);
    return 0;
}
