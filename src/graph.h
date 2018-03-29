#ifndef INF236_CA2_ADJMATRIX_H
#define INF236_CA2_ADJMATRIX_H

#include <stdlib.h>

#define maxMatrixSize 10000

typedef struct Graph {
    unsigned int size;
    long long** adjMatrix;

} Graph;


Graph* createGraph(unsigned int numberOfVertices){
    if(numberOfVertices > maxMatrixSize){
        return NULL;
    }
    Graph graph = {};
    graph.size = numberOfVertices;
    graph.adjMatrix = (long long**) malloc(numberOfVertices * sizeof(long long*));
    for(unsigned int i = 0; i < numberOfVertices; i++){
        graph.adjMatrix[i] = (long long*) malloc(numberOfVertices * sizeof(long long));
        for(unsigned int x = 0; x < numberOfVertices; x++){
            graph.adjMatrix[i][x] = 0;
        }
    }
    return &graph;
}

#endif //INF236_CA2_ADJMATRIX_H
