#include "bellmanFordCompleteGraphCpuParallel.h"

#define FLOAT_INFINITY 1000000

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


    float **localDistArr = (float **) malloc(sizeof(float *) * numberOfThreads);
    for (unsigned int i = 0; i < numberOfThreads; i++) {
        localDistArr[i] = (float *) malloc(sizeof(float) * graph->size);
    }

    omp_set_num_threads(numberOfThreads);
    starttime = omp_get_wtime();

    for (unsigned int n = 0; n < graph->size; n++) {

        for (unsigned int i = 0; i < numberOfThreads; i++) {
            int threadNum = omp_get_thread_num();
            memcpy(localDistArr[threadNum], graph->dist, graph->size);
        }

#pragma omp parallel for
        for (unsigned int y = 0; y < graph->size; y++) {
#pragma omp parallel for
            for (unsigned int x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                relax(graph->dist, weight, y, x);
                graph->predecessor[x] = y;
            }
        }

        for(unsigned int z = 0; z < graph->size;z++){
            float min = FLOAT_INFINITY;
            for(unsigned int i = 0; i < numberOfThreads; i++){
                if(localDistArr[i][z] < min){
                    graph->dist[z] = min;
                }
            }
        }
    }

    endtime = omp_get_wtime();
    if (localDistArr) {
        for (unsigned int i = 0; i < numberOfThreads; i++) {
            if (localDistArr[i]) {
                free(localDistArr[i]);
            }

        }
        free(localDistArr);
    }

    return endtime - starttime;
}
