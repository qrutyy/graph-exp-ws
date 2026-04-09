#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

GrB_Index find_max_out_degree_node(LAGraph_Graph G) {
    char msg[1024];
    LAGraph_Cached_OutDegree(G, msg);
    GrB_Index n;
    GrB_Vector_size(&n, G->out_degree);
    float *degrees = malloc(n * sizeof(float));
    GrB_Vector_extractTuples_FP32(NULL, degrees, &n, G->out_degree);
    GrB_Index max_node = 0;
    float max_deg = -1;
    for (GrB_Index i = 0; i < n; i++) {
        if (degrees[i] > max_deg) {
            max_deg = degrees[i];
            max_node = i;
        }
    }
    free(degrees);
    printf("DIAG: Selected Max Out-Degree Node: %llu (degree: %.0f)\n", max_node, max_deg);
    return max_node;
}

int main(int argc, char **argv) {
    if (argc < 5) return 1;

    char msg[1024];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    GrB_Matrix A = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) return 1;
    fclose(f);

    GrB_Type type;
    GxB_Matrix_type(&type, A);
    if (type == GrB_BOOL) {
        GrB_Index n, m;
        GrB_Matrix_nrows(&n, A); GrB_Matrix_ncols(&m, A);
        GrB_Matrix A2;
        GrB_Matrix_new(&A2, GrB_FP32, n, m);
        GrB_apply(A2, NULL, NULL, GrB_IDENTITY_FP32, A, NULL);
        GrB_free(&A); A = A2; type = GrB_FP32;
    }

    if (atoi(argv[3])) { // do_sym
        GrB_Index n; GrB_Matrix_nrows(&n, A);
        GrB_Matrix AT = NULL, S = NULL;
        GrB_Matrix_new(&AT, type, n, n);
        GrB_transpose(AT, NULL, NULL, A, NULL);
        GrB_Matrix_new(&S, type, n, n);
        GrB_BinaryOp op = (type == GrB_FP32) ? GrB_PLUS_FP32 : GrB_PLUS_FP64;
        GrB_Matrix_eWiseAdd_BinaryOp(S, NULL, NULL, op, A, AT, NULL);
        GrB_free(&A); GrB_free(&AT); A = S;
    }

    LAGraph_Graph G = NULL;
    LAGraph_Kind kind = atoi(argv[2]) ? LAGraph_ADJACENCY_DIRECTED : LAGraph_ADJACENCY_UNDIRECTED;
    LAGraph_New(&G, &A, kind, msg);
    
    GrB_Index source = find_max_out_degree_node(G);
    
    GrB_Scalar delta_scalar = NULL;
    GrB_Scalar_new(&delta_scalar, type);
    if (type == GrB_FP32) GrB_Scalar_setElement_FP32(delta_scalar, (float)atof(argv[4]));
    else GrB_Scalar_setElement_FP64(delta_scalar, atof(argv[4]));

    GrB_Vector dist = NULL;
    double t1 = LAGraph_WallClockTime();
    int result = LAGr_SingleSourceShortestPath(&dist, G, source, delta_scalar, msg);
    double t2 = LAGraph_WallClockTime();
    
    if (result == GrB_SUCCESS) {
        GrB_Index reached; GrB_Vector_nvals(&reached, dist);
        printf("DIAG: SSSP Reached nodes: %llu\n", reached);
        printf("TIME: %f\n", t2 - t1); // Строгий формат для парсинга
    }

    GrB_free(&dist); GrB_free(&delta_scalar);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}

