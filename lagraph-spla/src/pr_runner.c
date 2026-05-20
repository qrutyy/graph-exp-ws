#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_market_file> [niters_max]\n", argv[0]);
        return 1;
    }

    char msg[1024];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    GrB_Matrix A = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) {
        printf("Failed to read matrix\n");
        return 1;
    }
    fclose(f);

	LAGraph_Graph G = NULL;
    LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);

    LAGraph_Cached_OutDegree(G, msg);
    LAGraph_Cached_AT(G, msg);

    float alpha = 0.85f;
    float eps = 1e-4f;
    int itermax = (argc > 2) ? atoi(argv[2]) : 100;
    int iters_taken = 0;
    GrB_Vector centrality = NULL;

    double t1 = LAGraph_WallClockTime();
    int result = LAGr_PageRank(&centrality, &iters_taken, G, alpha, eps, itermax, msg);
    double t2 = LAGraph_WallClockTime();

    if (result == GrB_SUCCESS) {
        double total_time_ms = (t2 - t1) * 1000.0;
        printf("RESULT_SUCCESS\n");
        printf("ITERS: %d\n", iters_taken);
        printf("TOTAL_TIME_MS: %f\n", total_time_ms);
        printf("MS_PER_ITER: %f\n", total_time_ms / iters_taken);
    } else {
        printf("RESULT_FAILED: %s\n", msg);
    }

    GrB_free(&centrality);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
