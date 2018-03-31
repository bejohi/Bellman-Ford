#include "bellmanFordCompleteGraphCpuParallel.h"

static inline float compareAndSwap(float *ptr, float oldValue, float newValue) {
    float tmp = *ptr;
    if (*ptr == oldValue || *ptr > newValue) {
        *ptr = newValue;
    }
    return tmp;
}

static inline void relax(float *distanceArray, float weight, unsigned int vertex1, unsigned int vertex2) {
    if (distanceArray[vertex1] + weight < distanceArray[vertex2]) {
        compareAndSwap(&distanceArray[vertex2], distanceArray[vertex2], distanceArray[vertex1] + weight);
    }
}


double bellmanFordParallelCpu(CompleteGraph *graph, unsigned int startVertex, unsigned int numberOfThreads) {
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }
    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;


    omp_set_num_threads(numberOfThreads);
    bool finished;
    starttime = omp_get_wtime();
    for (unsigned int n = 0; n < graph->size; n++) {
        finished = true;
#pragma omp parallel for
        for (unsigned int y = 0; y < graph->size; y++) {
            for (unsigned int x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                relax(graph->dist, weight, y, x);
                graph->predecessor[x] = y;
                finished = false;
            }
        }
        if(finished){
            break;
        }
    }

    endtime = omp_get_wtime();
    return endtime - starttime;
}
