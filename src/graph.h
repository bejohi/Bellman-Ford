#ifndef INF236_CA2_GRAPH_H
#define INF236_CA2_GRAPH_H

#include <stdlib.h>

#define EDGE_NOT_INIT (-1)

// TODO: Check suspicious casting from long long to unsigned long

typedef struct Graph {
    unsigned long numberOfVertices;
    unsigned long numberOfEdges;
    unsigned long edgeListSize;
    unsigned long edgePointer;
    long long *edgeList;
} Graph;

Graph createGraph(unsigned long numberOfVertices, unsigned long numberOfEdges);

void addEdge(Graph *graph, unsigned long vertex1, unsigned long vertex2, long long weight);

void destroyGraph(Graph *graph);

#endif //INF236_CA2_GRAPH_H
