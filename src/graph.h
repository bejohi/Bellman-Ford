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
}

void addEdgeInBothDirections(Graph *graph, ULL edge1, ULL edge2, double weight){

    // TODO Bounds check

    Edge* currentEdge = &graph->adjList[edge1];
    while(true){
        if(currentEdge->next != NULL){
            currentEdge = currentEdge->next;
            continue;
        }
        currentEdge->next = malloc(sizeof(Edge));
        currentEdge->next->index = edge2;
        currentEdge->next->next = NULL;
        currentEdge->next->weight = weight;
        break;
    }
    currentEdge = &graph->adjList[edge2];
    while(true){
        if(currentEdge->next != NULL){
            currentEdge = currentEdge->next;
            continue;
        }
        currentEdge->next = malloc(sizeof(Edge));
        currentEdge->next->index = edge1;
        currentEdge->next->next = NULL;
        currentEdge->next->weight = weight;
        break;
    }
}

void destroyGraph(Graph *graph) {
    free(graph->adjList);
    // TODO: Free all edges
}

#endif //INF236_CA2_GRAPH_H
