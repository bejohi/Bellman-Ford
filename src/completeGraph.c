#include "completeGraph.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000

inline void initArrays(float *distanceArray, unsigned int *prevArray, long size) {
    for (unsigned long i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
        prevArray[i] = NO_PREV;
    }
}

CompleteGraph createCompleteGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    CompleteGraph completeGraph = {.size = size, .isDirected = false, .error = false};

    completeGraph.dist = (float *) malloc(sizeof(float) * size);
    completeGraph.predecessor = (unsigned int *) malloc(sizeof(unsigned int) * size);
    completeGraph.adjMatrix = (float **) malloc(sizeof(float *) * size);

    if (!completeGraph.dist || !completeGraph.predecessor || !completeGraph.adjMatrix) {
        destroyCompleteGraph(&completeGraph);
        return (CompleteGraph) {.error = true};
    }

    unsigned int i, x;

    for (i = 0; i < size; i++) {
        completeGraph.adjMatrix[i] = (float *) malloc(sizeof(float) * size);
        if (!completeGraph.adjMatrix[i]) {
            destroyCompleteGraph(&completeGraph);
            return (CompleteGraph) {.error = true};
        }
        if (i == 0) {
            for (x = 0; x < size; x++) {
                completeGraph.adjMatrix[i][x] = 0;
            }
        } else {
            memcpy(completeGraph.adjMatrix[i], completeGraph.adjMatrix[0], sizeof(float) * size);
        }

    }
    return completeGraph;
}

void addEdgeCompleteGraph(CompleteGraph *graph, unsigned int startVertex, unsigned int endVertex, float weight) {
    if (!graph) {
        return;
    }
    if (!graph->adjMatrix || endVertex >= graph->size || startVertex >= graph->size) {
        graph->error = true;
        return;
    }

    graph->adjMatrix[startVertex][endVertex] = weight;
    if (graph->isDirected) {
        graph->adjMatrix[endVertex][startVertex] = weight;
    }

}

void destroyCompleteGraph(CompleteGraph *completeGraph) {
    free(completeGraph->predecessor);
    free(completeGraph->dist);
    unsigned int i;
    for (i = 0; i < completeGraph->size; i++) {
        if (completeGraph->adjMatrix[i]) {
            free(completeGraph->adjMatrix[i]);
        }
    }
    free(completeGraph->adjMatrix);
}



