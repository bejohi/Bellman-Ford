#ifndef INF236_CA2_GRAPH_H
#define INF236_CA2_GRAPH_H

#include <stdlib.h>

#define EDGE_NOT_INIT (-1)

// TODO: Check suspicious casting from long long to unsigned long

typedef struct Graph {
    long numberOfVertices;
    long numberOfEdges;
    long edgeListSize;
    long edgePointer;
    float *edgeList;
} Graph;

Graph createGraph(long numberOfVertices, long numberOfEdges);

void addEdge(Graph *graph, long vertex1, long vertex2, float weight);

void destroyGraph(Graph *graph);

#endif //INF236_CA2_GRAPH_H
