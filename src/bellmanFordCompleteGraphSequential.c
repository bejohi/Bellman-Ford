#include "bellmanFordCompleteGraphSequential.h"

double bellmanFord(CompleteGraph *graph, unsigned int startVertex) {
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }
    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    time_t startTime, endTime;
    bool finished;
    unsigned int n, y, x;
    time(&startTime);
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
    time(&endTime);
    return difftime(endTime,startTime);
}