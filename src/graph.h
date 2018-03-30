#ifndef INF236_CA2_GRAPH_H
#define INF236_CA2_GRAPH_H


#include <stdlib.h>

#define EDGE_NOT_INIT (-1)

typedef struct Graph {
    unsigned long numberOfVertices;
    unsigned long numberOfEdges;
    unsigned long edgeListSize;
    long long *edgeList;
} Graph;

Graph createGraph(unsigned long numberOfVertices, unsigned long numberOfEdges) {
    Graph graph = {};
    graph.numberOfEdges = numberOfEdges;
    graph.numberOfVertices = numberOfVertices;
    graph.edgeListSize = numberOfEdges * 3;
    graph.edgeList = (long long *) malloc(sizeof(long) * graph.edgeListSize);
    for(unsigned long i = 0; i < graph.edgeListSize; i++){
        graph.edgeList[i] = EDGE_NOT_INIT;
    }

    return graph;
}

void destroyGraph(Graph *graph) {
    free(graph->edgeList);
}

#endif //INF236_CA2_GRAPH_H
