#include "bellmanFordCompleteGraphSequential.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000

inline static void initArrays(float *distanceArray, unsigned int* prevArray, long size){
    for(unsigned long i = 0; i < size; i++){
        distanceArray[i] = INFINIT_DISTANCE;
        prevArray[i] = NO_PREV;
    }
}

double bellmanFord(CompleteGraph* graph, unsigned int startVertex){
    if(!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist){
        return -1;
    }
    initArrays(graph->dist,graph->predecessor,graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    starttime = omp_get_wtime();
    for(unsigned int n = 0; n < graph->size; n++){
        for(unsigned int y = 0; y < graph->size; y++){
            for(unsigned int x = 0; x < graph->size; x++){
                float weight = graph->adjMatrix[y][x];
                if(graph->dist[y] + weight < graph->dist[x]){
                    graph->dist[x] = graph->dist[y] + weight;
                    graph->predecessor[x] = y;
                }
            }
        }
    }
    endtime = omp_get_wtime();
    return endtime - starttime;
}