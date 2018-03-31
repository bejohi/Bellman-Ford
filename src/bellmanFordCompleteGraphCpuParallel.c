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
        distanceArray[vertex2] = distanceArray[vertex1] + weight;
    }
}


double bellmanFordParallelCpu(CompleteGraph *graph, unsigned int startVertex, unsigned int numberOfThreads) {
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }
    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;

    register float **matrix = graph->adjMatrix;
    register unsigned int size = graph->size;
    register float *dist = graph->dist;
    omp_set_num_threads(numberOfThreads);
    bool finished;
    starttime = omp_get_wtime();
    for (unsigned int n = 0; n < size; n++) {
        finished = true;
#pragma omp parallel for
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int x = 0; x < size; x++) {
                float weight = matrix[y][x];
                relax(dist, weight, y, x);
                //graph->predecessor[x] = y;
                finished = false;
            }
        }
        if (finished) {
            break;
        }
    }

    endtime = omp_get_wtime();
    return endtime - starttime;
}
