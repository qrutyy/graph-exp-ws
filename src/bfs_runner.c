#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <kind: 0=Undir, 1=Dir> <symmetrize: 0=No, 1=Yes>\n", argv[0]);
        return 1;
    }

    char msg[1024];
    if (LAGraph_Init(msg) != GrB_SUCCESS) return 1;
//	GxB_Global_Option_set(GxB_BURBLE, true);
    int nthreads = omp_get_max_threads(); 
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, nthreads);

	int actual_nthreads;
	GxB_Global_Option_get(GxB_GLOBAL_NTHREADS, &actual_nthreads);
	printf("GraphBLAS threads set to: %d\n", actual_nthreads);

    char *filename = argv[1];
    int is_directed = atoi(argv[2]);
    int do_sym = atoi(argv[3]);

    GrB_Matrix A = NULL;
    FILE *f = fopen(filename, "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) {
        fprintf(stderr, "Read failed\n");
        return 1;
    }
    fclose(f);

    if (do_sym) {
        GrB_Index n;
        GrB_Matrix_nrows(&n, A);
        GrB_Matrix AT = NULL, S = NULL;
        GrB_Matrix_new(&AT, GrB_BOOL, n, n);
        GrB_transpose(AT, NULL, NULL, A, NULL);
        GrB_Matrix_new(&S, GrB_BOOL, n, n);
        GrB_Matrix_eWiseAdd_BinaryOp(S, NULL, NULL, GrB_LOR, A, AT, NULL);
        GrB_free(&A); GrB_free(&AT);
        A = S;
        is_directed = 0; // undirected 
    }

    LAGraph_Graph G = NULL;
    LAGraph_Kind kind = is_directed ? LAGraph_ADJACENCY_DIRECTED : LAGraph_ADJACENCY_UNDIRECTED;
    LAGraph_New(&G, &A, kind, msg);
    LAGraph_Cached_OutDegree(G, msg);

    GrB_Vector parent = NULL, level = NULL;
    double t1 = LAGraph_WallClockTime();
    int result = LAGr_BreadthFirstSearch(&level, &parent, G, 0, msg);
    double t2 = LAGraph_WallClockTime();

	GrB_Index nvisited ;
	GrB_Vector_nvals(&nvisited, level);
	printf("Visited nodes: %llu\n", nvisited);

    if (result == GrB_SUCCESS) {
        printf("cpu(s): %f\n", t2 - t1);
    }

    GrB_free(&parent); GrB_free(&level);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
