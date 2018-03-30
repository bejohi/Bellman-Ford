#include "graph.h"


Graph createGraph(unsigned long numberOfVertices, unsigned long numberOfEdges) {
    Graph graph = {};
    graph.numberOfEdges = numberOfEdges;
    graph.numberOfVertices = numberOfVertices;
    graph.edgeListSize = numberOfEdges * 3;
    graph.edgeList = (long long *) malloc(sizeof(long long) * graph.edgeListSize);
    for (unsigned long i = 0; i < graph.edgeListSize; i++) {
        graph.edgeList[i] = EDGE_NOT_INIT;
    }
    graph.edgePointer = 0;

    return graph;
}

void addEdge(Graph *graph, unsigned long vertex1, unsigned long vertex2, long long weight) {
    graph->edgeList[graph->edgePointer] = (long long) vertex1;
    graph->edgeList[graph->edgePointer+1] = (long long) vertex2;
    graph->edgeList[graph->edgePointer+2] = weight;
    graph->edgePointer += 3;

}

void destroyGraph(Graph *graph) {
    if(!graph || !graph->edgeList){
        return;
    }
    free(graph->edgeList);
}