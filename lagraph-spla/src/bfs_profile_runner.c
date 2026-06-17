#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"

/*
 * BFS per-level profiling runner.
 *
 * Implements push-only BFS using GrB_vxm (frontier * A) with per-level timing.
 * Reports: level, frontier_size, time_ms for each BFS level.
 *
 * Usage: bfs_profile_runner <matrix.mtx> [source_node]
 * Output CSV (stdout): level,frontier_size,time_ms,visited_total
 */

static GrB_Index find_max_out_degree_node(LAGraph_Graph G)
{
    GrB_Index n = 0;
    GrB_Matrix_nrows(&n, G->A);
    GrB_Index *node_ids = (GrB_Index *)malloc(n * sizeof(GrB_Index));
    int64_t   *degrees  = (int64_t  *)malloc(n * sizeof(int64_t));
    GrB_Index  nvals    = n;

    GrB_Vector_extractTuples_INT64(node_ids, degrees, &nvals, G->out_degree);

    GrB_Index best_node = 0;
    int64_t   best_deg  = -1;
    for (GrB_Index i = 0; i < nvals; i++) {
        if (degrees[i] > best_deg) {
            best_deg  = degrees[i];
            best_node = node_ids[i];
        }
    }
    free(node_ids); free(degrees);
    return best_node;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx> [source]\n", argv[0]);
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

    /* convert to BOOL for BFS */
    GrB_Matrix B = NULL;
    GrB_Matrix_new(&B, GrB_BOOL, n, n);
    GrB_Matrix_assign_BOOL(B, A, NULL, true, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S);
    GrB_free(&A);

    LAGraph_Graph G = NULL;
    LAGraph_New(&G, &B, LAGraph_ADJACENCY_DIRECTED, msg);
    LAGraph_Cached_OutDegree(G, msg);

    GrB_Index source = (argc > 2) ? (GrB_Index)atol(argv[2])
                                   : find_max_out_degree_node(G);
    fprintf(stderr, "source=%llu\n", (unsigned long long)source);

    /* ---- BFS vectors ---- */
    GrB_Vector frontier = NULL, visited = NULL, new_front = NULL;
    GrB_Vector_new(&frontier,  GrB_BOOL, n);
    GrB_Vector_new(&visited,   GrB_BOOL, n);
    GrB_Vector_new(&new_front, GrB_BOOL, n);

    /* set source */
    GrB_Vector_setElement_BOOL(frontier, true, source);
    GrB_Vector_setElement_BOOL(visited,  true, source);
    GrB_wait(frontier, GrB_MATERIALIZE);
    GrB_wait(visited,  GrB_MATERIALIZE);

    /* ---- descriptor: complement of visited as mask ---- */
    GrB_Descriptor desc_rc = NULL;
    GrB_Descriptor_new(&desc_rc);
    GrB_Descriptor_set(desc_rc, GrB_MASK, GrB_COMP);   /* complement mask */
    GrB_Descriptor_set(desc_rc, GrB_MASK, GrB_STRUCTURE); /* struct only */

    /* Actually use replace + complement-mask descriptor */
    GrB_Descriptor desc_crs = NULL;
    GrB_Descriptor_new(&desc_crs);
    GrB_Descriptor_set(desc_crs, GrB_MASK, GrB_COMP | GrB_STRUCTURE);
    GrB_Descriptor_set(desc_crs, GrB_OUTP, GrB_REPLACE);

    printf("level,frontier_size,time_ms,visited_total\n");

    int level = 0;
    GrB_Index total_visited = 1;

    while (1) {
        GrB_Index front_size = 0;
        GrB_Vector_nvals(&front_size, frontier);
        if (front_size == 0) break;
        level++;

        /* new_front = frontier * B,  mask = ~visited (structural complement) */
        double t1 = LAGraph_WallClockTime();
        GrB_vxm(new_front, visited, NULL, GrB_LOR_LAND_SEMIRING_BOOL,
                frontier, G->A, desc_crs);
        GrB_wait(new_front, GrB_MATERIALIZE);
        double t2 = LAGraph_WallClockTime();

        GrB_Index new_size = 0;
        GrB_Vector_nvals(&new_size, new_front);

        printf("%d,%llu,%.4f,%llu\n",
               level,
               (unsigned long long)front_size,
               (t2 - t1) * 1000.0,
               (unsigned long long)total_visited);
        fflush(stdout);

        if (new_size == 0) break;

        /* visited |= new_front */
        GrB_eWiseAdd(visited, NULL, NULL, GrB_LOR, visited, new_front, NULL);
        GrB_wait(visited, GrB_MATERIALIZE);

        /* swap frontier <- new_front, clear new_front */
        GrB_Vector tmp = frontier; frontier = new_front; new_front = tmp;
        GrB_Vector_clear(new_front);

        total_visited += new_size;
    }

    fprintf(stderr, "levels=%d  visited=%llu / %llu\n",
            level, (unsigned long long)total_visited, (unsigned long long)n);

    GrB_free(&frontier); GrB_free(&visited); GrB_free(&new_front);
    GrB_free(&desc_rc); GrB_free(&desc_crs);
    LAGraph_Delete(&G, msg);
    LAGraph_Finalize(msg);
    return 0;
}
