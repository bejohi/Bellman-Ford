#include "bellmanFordCompleteGraphSequential.h"

double currentTime(){
    return omp_get_wtime();
}

double bellmanFord(CompleteGraph *graph, unsigned int startVertex) {
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }
    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    double startTime;
    bool finished;
    unsigned int n, y, x;
    startTime = currentTime();
    for (n = 0; n < graph->size; n++) {
        finished = true;
        for (y = 0; y < graph->size; y++) {
            for (x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                if (graph->dist[y] + weight < graph->dist[x]) {
                    graph->dist[x] = graph->dist[y] + weight;
                    graph->predecessor[x] = y;
                    finished = false;
                }
            }
        }
        if (finished) {
            break;
        }
    }
    return currentTime() - startTime;
}