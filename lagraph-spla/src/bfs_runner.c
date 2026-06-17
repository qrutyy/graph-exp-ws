#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/* Find the node with the highest out-degree.  Returns 0 on failure. */
static GrB_Index find_max_out_degree_node(LAGraph_Graph G) {
    char msg[1024];
    LAGraph_Cached_OutDegree(G, msg);

    GrB_Index nvals;
    GrB_Vector_nvals(&nvals, G->out_degree);

    GrB_Index *node_ids = malloc(nvals * sizeof(GrB_Index));
    int64_t   *degrees  = malloc(nvals * sizeof(int64_t));

    GrB_Vector_extractTuples_INT64(node_ids, degrees, &nvals, G->out_degree);

    GrB_Index max_node = 0;
    int64_t   max_deg  = -1;
    for (GrB_Index i = 0; i < nvals; i++) {
        if (degrees[i] > max_deg) {
            max_deg  = degrees[i];
            max_node = node_ids[i];
        }
    }
    free(node_ids);
    free(degrees);
    printf("DIAG: source=%llu out_degree=%lld\n", max_node, max_deg);
    return max_node;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <matrix.mtx> <kind: 0=Undir, 1=Dir> <symmetrize: 0=No, 1=Yes>\n",
            argv[0]);
        return 1;
    }

    char msg[1024];
    if (LAGraph_Init(msg) != GrB_SUCCESS) return 1;

    int nthreads = omp_get_max_threads();
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, nthreads);
    int actual_nthreads;
    GxB_Global_Option_get(GxB_GLOBAL_NTHREADS, &actual_nthreads);
    printf("GraphBLAS threads set to: %d\n", actual_nthreads);

    char *filename  = argv[1];
    int   is_directed = atoi(argv[2]);
    int   do_sym    = atoi(argv[3]);

    /* ---- Load matrix ---------------------------------------------------- */
    GrB_Matrix A = NULL;
    FILE *f = fopen(filename, "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) {
        fprintf(stderr, "Read failed\n");
        return 1;
    }
    fclose(f);

    /* ---- Optional symmetrisation ---------------------------------------- */
    if (do_sym) {
        GrB_Index n;
        GrB_Matrix_nrows(&n, A);
        GrB_Matrix AT = NULL, S = NULL;
        GrB_Matrix_new(&AT, GrB_BOOL, n, n);
        GrB_transpose(AT, NULL, NULL, A, NULL);
        GrB_Matrix_new(&S, GrB_BOOL, n, n);
        GrB_eWiseAdd(S, NULL, NULL, GrB_LOR, A, AT, NULL);
        GrB_free(&A); GrB_free(&AT);
        A = S;
        is_directed = 0;
    }

    /* ---- Build LAGraph graph -------------------------------------------- */
    LAGraph_Graph G = NULL;
    LAGraph_Kind kind = is_directed
        ? LAGraph_ADJACENCY_DIRECTED
        : LAGraph_ADJACENCY_UNDIRECTED;
    LAGraph_New(&G, &A, kind, msg);

    /* Cache properties.
       For directed graphs we must cache AT so that LAGr_BreadthFirstSearch
       can use the direction-optimising (push+pull) strategy.  Without AT the
       algorithm silently falls back to push-only, which is significantly
       slower on graphs with a dense neighbourhood. */
    LAGraph_Cached_OutDegree(G, msg);
    if (is_directed) {
        LAGraph_Cached_AT(G, msg);  /* required for direction-optimising BFS */
    }

    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, G->A);
    printf("nnz=%llu kind=%s\n", nvals, is_directed ? "directed" : "undirected");

    GrB_Index source = find_max_out_degree_node(G);

    /* ---- Run BFS --------------------------------------------------------- */
    GrB_Vector parent = NULL, level = NULL;
    double t1 = LAGraph_WallClockTime();
    int result = LAGr_BreadthFirstSearch(&level, &parent, G, source, msg);
    double t2 = LAGraph_WallClockTime();

    if (result != GrB_SUCCESS) {
        fprintf(stderr, "BFS failed: %s\n", msg);
        return 1;
    }

    GrB_Index nvisited;
    GrB_Vector_nvals(&nvisited, level);
    printf("DIAG: visited=%llu\n", nvisited);
    printf("cpu(s): %f\n", t2 - t1);

    GrB_free(&parent); GrB_free(&level);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
