#include "bellmanFordCompleteGraphCpuParallel.h"


double bellmanFordParallelCpu(CompleteGraph *graph, unsigned int startVertex, unsigned int numberOfThreads) {
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }
    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    starttime = omp_get_wtime();

    omp_set_num_threads(numberOfThreads);

    for (unsigned int n = 0; n < graph->size; n++) {
        for (unsigned int y = 0; y < graph->size; y++) {
#pragma omp parallel for
            for (unsigned int x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                if (graph->dist[y] + weight < graph->dist[x]) {
                    graph->dist[x] = graph->dist[y] + weight;
                    graph->predecessor[x] = y;
                }
            }
        }
    }
    endtime = omp_get_wtime();
    return endtime - starttime;
}
