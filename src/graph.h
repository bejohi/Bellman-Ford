#ifndef INF236_CA2_GRAPH_H
#define INF236_CA2_GRAPH_H

#include <stdlib.h>
#include <stdio.h>

typedef unsigned long long ULL;

typedef struct Edge {
    ULL index;
    double weight;
    struct Edge *next;
} Edge;

typedef struct {
    ULL size;
    Edge *adjList;
} Graph;

void initGraph(Graph *graph, ULL graphSize) {
    graph->size = graphSize;
    graph->adjList = (Edge *) malloc(sizeof(Edge) * graphSize);
    for (ULL i = 0; i < graphSize; i++) {
        graph->adjList[i].next = NULL;
    }
}

void destroyGraph(Graph *graph) {
    free(graph->adjList);
    graph = NULL;
}

#endif //INF236_CA2_GRAPH_H
