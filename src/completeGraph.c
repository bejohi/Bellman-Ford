#include "completeGraph.h"

CompleteGraph createCompleteGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    CompleteGraph completeGraph = {.size = size, .isDirected = true};

    completeGraph.dist = (float *) malloc(sizeof(float) * size);
    completeGraph.predecessor = (unsigned int*) malloc(sizeof(unsigned int) * size);

    completeGraph.adjMatrix = (float **) malloc(sizeof(float *) * size);
    for (unsigned int i = 0; i < size; i++) {
        completeGraph.adjMatrix[i] = (float *) malloc(sizeof(float) * size);
        for (unsigned int x = 0; x < size; x++) {
            completeGraph.adjMatrix[i][x] = 0;
        }
    }
    return completeGraph;
}

void destroyCompleteGraph(CompleteGraph* completeGraph){
    if(!completeGraph){
        return;
    }
    if(completeGraph->predecessor){
        free(completeGraph->predecessor);
    }
    if(completeGraph->dist){
        free(completeGraph->dist);
    }
    if(completeGraph->adjMatrix){
        for(unsigned int i = 0; i < completeGraph->size; i++){
            if(completeGraph->adjMatrix[i]){
                free(completeGraph->adjMatrix[i]);
            }
        }
        free(completeGraph->adjMatrix);
    }
}

