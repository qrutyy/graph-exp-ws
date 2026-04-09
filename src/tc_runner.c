#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_path> <method_idx> <presort_idx>\n", argv[0]);
        printf("Methods: 1:Burkhardt, 2:Cohen, 3:Sandia_LL, 4:Sandia_UU, 5:Sandia_LUT, 6:Sandia_ULT\n");
        printf("Presort: 0:None, 1:Ascending, 2:Descending\n");
        return 1;
    }

    char msg[1024];
    LAGraph_Init(msg);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, omp_get_max_threads());

    GrB_Matrix A = NULL;
    FILE *f = fopen(argv[1], "r");
    if (!f || LAGraph_MMRead(&A, f, msg) != GrB_SUCCESS) return 1;
    fclose(f);

    LAGraph_Graph G = NULL;
    if (LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg) != GrB_SUCCESS) return 1;
    if (LAGraph_DeleteSelfEdges(G, msg) != GrB_SUCCESS) return 1;
    if (LAGraph_Cached_OutDegree(G, msg) != GrB_SUCCESS) return 1;

    // Выбор метода
    int m_idx = atoi(argv[2]);
    LAGr_TriangleCount_Method method;
    const char *m_name;
    switch(m_idx) {
        case 1: method = LAGr_TriangleCount_Burkhardt; m_name = "Burkhardt"; break;
        case 2: method = LAGr_TriangleCount_Cohen; m_name = "Cohen"; break;
        case 3: method = LAGr_TriangleCount_Sandia_LL; m_name = "Sandia_LL"; break;
        case 4: method = LAGr_TriangleCount_Sandia_UU; m_name = "Sandia_UU"; break;
        case 5: method = LAGr_TriangleCount_Sandia_LUT; m_name = "Sandia_LUT"; break;
        case 6: method = LAGr_TriangleCount_Sandia_ULT; m_name = "Sandia_ULT"; break;
        default: method = LAGr_TriangleCount_AutoMethod; m_name = "Auto"; break;
    }

    // Выбор сортировки
    int s_idx = atoi(argv[3]);
    LAGr_TriangleCount_Presort presort;
    const char *s_name;
    switch(s_idx) {
        case 0: presort = LAGr_TriangleCount_NoSort; s_name = "NoSort"; break;
        case 1: presort = LAGr_TriangleCount_Ascending; s_name = "Asc"; break;
        case 2: presort = LAGr_TriangleCount_Descending; s_name = "Desc"; break;
        default: presort = LAGr_TriangleCount_AutoSort; s_name = "AutoSort"; break;
    }

    uint64_t ntriangles = 0;
    double t1 = LAGraph_WallClockTime();
    int result = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    double t2 = LAGraph_WallClockTime();

    if (result == GrB_SUCCESS) {
        printf("RESULT_SUCCESS|METHOD:%s|SORT:%s\n", m_name, s_name);
        printf("TRIANGLES: %llu\n", ntriangles);
        printf("TIME: %f\n", t2 - t1);
    } else {
        printf("FAILED: %s\n", msg);
    }

    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
