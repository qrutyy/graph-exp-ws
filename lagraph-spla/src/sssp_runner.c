#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/* Find the node with the highest out-degree — used as SSSP source so that
   we start from a well-connected vertex and measure a meaningful traversal. */
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
    printf("DIAG: Selected Max Out-Degree Node: %llu (degree: %lld)\n", max_node, max_deg);
    return max_node;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <matrix.mtx> <kind: 0=Undir, 1=Dir> <symmetrize: 0=No, 1=Yes> <delta>\n",
            argv[0]);
        return 1;
    }

    char msg[1024];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    /* ---- Load matrix ---------------------------------------------------- */
    GrB_Matrix A = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) return 1;
    fclose(f);

    /* SSSP requires weighted edges; convert boolean pattern to float weights.
       All edges receive weight 1.0 (unit-weight shortest paths). */
    GrB_Type type;
    GxB_Matrix_type(&type, A);
    if (type == GrB_BOOL) {
        GrB_Index n, m;
        GrB_Matrix_nrows(&n, A);
        GrB_Matrix_ncols(&m, A);
        GrB_Matrix A2;
        GrB_Matrix_new(&A2, GrB_FP32, n, m);
        GrB_apply(A2, NULL, NULL, GrB_IDENTITY_FP32, A, NULL);
        GrB_free(&A); A = A2; type = GrB_FP32;
    }

    int is_directed = atoi(argv[2]);
    int do_sym      = atoi(argv[3]);

    /* ---- Optional symmetrisation ---------------------------------------- */
    if (do_sym) {
        GrB_Index n;
        GrB_Matrix_nrows(&n, A);
        GrB_Matrix AT = NULL, S = NULL;
        GrB_Matrix_new(&AT, type, n, n);
        GrB_transpose(AT, NULL, NULL, A, NULL);
        GrB_Matrix_new(&S, type, n, n);
        /* Use MIN so that when both directions exist, the shorter edge wins. */
        GrB_BinaryOp op = (type == GrB_FP32) ? GrB_MIN_FP32 : GrB_MIN_FP64;
        GrB_eWiseAdd(S, NULL, NULL, op, A, AT, NULL);
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

    /* Source selection: highest out-degree node.  This is the same heuristic
       used in the LAGraph benchmark suite — it maximises reachable nodes and
       makes timing results more reproducible across graph types. */
    GrB_Index source = find_max_out_degree_node(G);

    /* ---- Set up delta-stepping parameter --------------------------------- */
    GrB_Scalar delta_scalar = NULL;
    GrB_Scalar_new(&delta_scalar, type);
    if (type == GrB_FP32)
        GrB_Scalar_setElement_FP32(delta_scalar, (float)atof(argv[4]));
    else
        GrB_Scalar_setElement_FP64(delta_scalar, atof(argv[4]));

    printf("DIAG: Running SSSP delta=%s source=%llu kind=%s\n",
           argv[4], source, is_directed ? "directed" : "undirected");

    /* ---- Run SSSP -------------------------------------------------------- */
    GrB_Vector dist = NULL;
    double t1 = LAGraph_WallClockTime();
    int result = LAGr_SingleSourceShortestPath(&dist, G, source, delta_scalar, msg);
    double t2 = LAGraph_WallClockTime();

    if (result == GrB_SUCCESS) {
        GrB_Index reached;
        GrB_Vector_nvals(&reached, dist);
        printf("DIAG: SSSP Reached nodes: %llu\n", reached);
        printf("TIME: %f\n", t2 - t1);
    } else {
        printf("DIAG: SSSP Failed: %s\n", msg);
    }

    GrB_free(&dist);
    GrB_free(&delta_scalar);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
