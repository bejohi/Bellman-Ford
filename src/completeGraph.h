#ifndef INF236_CA2_COMPLETEGRAPH_H
#define INF236_CA2_COMPLETEGRAPH_H

#include <stdlib.h>
#include <stdbool.h>
#include <float.h>

#define MAX_GRAPH_SIZE 10000

typedef struct CompleteGraph {
    unsigned int size;
    bool isDirected;
    bool error;
    float **adjMatrix;
    float *dist;
    unsigned int *predecessor;
} CompleteGraph;


void destroyCompleteGraph(CompleteGraph *completeGraph);

CompleteGraph createCompleteGraph(unsigned int size);

void addEdgeCompleteGraph(CompleteGraph *graph, unsigned int startVertex, unsigned int endVertex, float weight);

#endif //INF236_CA2_COMPLETEGRAPH_H
