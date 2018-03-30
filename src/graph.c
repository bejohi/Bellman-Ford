#include "graph.h"


Graph createGraph(long numberOfVertices, long numberOfEdges) {
    Graph graph = {};
    graph.numberOfEdges = numberOfEdges;
    graph.numberOfVertices = numberOfVertices;
    graph.edgeListSize = numberOfEdges * 3;
    graph.edgeList = (float *) malloc(sizeof(float) * graph.edgeListSize);
    for (unsigned long i = 0; i < graph.edgeListSize; i++) {
        graph.edgeList[i] = EDGE_NOT_INIT;
    }
    graph.edgePointer = 0;

    return graph;
}

void addEdge(Graph *graph, long vertex1, long vertex2, float weight) {
    graph->edgeList[graph->edgePointer] = vertex1;
    graph->edgeList[graph->edgePointer+1] = vertex2;
    graph->edgeList[graph->edgePointer+2] = weight;
    graph->edgePointer += 3;

}

void destroyGraph(Graph *graph) {
    if(!graph || !graph->edgeList){
        return;
    }
    free(graph->edgeList);
}