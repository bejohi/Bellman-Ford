#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "graph.h"
#include <float.h>

#define NO_PREDECESSOR (-1)

static void initGraphArrays(double* distanceArray, long* preArray, unsigned int length){
    for(unsigned int i = 0; i < length; i++){
        distanceArray[i] = DBL_MAX;
        preArray[i] = NO_PREDECESSOR;
    }
}


void bellmanFord(Graph* graph, unsigned int startVertex, double* distanceArray){
    long* predecessor = malloc(sizeof(long) * graph->size);
    initGraphArrays(distanceArray,predecessor,graph->size);
    distanceArray[startVertex] = 0;

    for(unsigned int n = 0; n < graph->size; n++){
        // TODO: Add Relaxation Operation.
    }

    free(predecessor);
}