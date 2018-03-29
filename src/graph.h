#ifndef INF236_CA2_ADJMATRIX_H
#define INF236_CA2_ADJMATRIX_H

#include <stdlib.h>
#include <math.h>

#define maxMatrixSize 10000
#define noEdge -0

typedef struct Graph {
    unsigned int size;
    double **adjMatrix;
    bool isBidirectional;

} Graph;


Graph createGraph(unsigned int numberOfVertices) {
    if (numberOfVertices > maxMatrixSize) {
        numberOfVertices = maxMatrixSize;
    }
    Graph graph = {};
    graph.size = numberOfVertices;
    graph.adjMatrix = (double **) malloc(numberOfVertices * sizeof(double *));
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        graph.adjMatrix[i] = (double *) malloc(numberOfVertices * sizeof(double));
        for (unsigned int x = 0; x < numberOfVertices; x++) {
            graph.adjMatrix[i][x] = noEdge;
        }
    }
    graph.isBidirectional = true;
    return graph;
}

void destroyGraph(Graph *graph) {
    for (unsigned int i = 0; i < graph->size; i++) {
        if (graph->adjMatrix[i] != NULL) {
            free(graph->adjMatrix[i]);
        }
    }
    if (graph->adjMatrix != NULL) {
        free(graph->adjMatrix);
    }
}

void addEdge(Graph *graph, unsigned int vertex1, unsigned int vertex2, double weight) {
    if (graph->size <= vertex1 || graph->size <= vertex2) {
        return;
    }
    graph->adjMatrix[vertex1][vertex2] = weight;

    if (graph->isBidirectional) {
        graph->adjMatrix[vertex2][vertex1] = weight;
    }
}


#endif //INF236_CA2_ADJMATRIX_H
