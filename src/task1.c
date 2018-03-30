#include "task1.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 10000000
#define NO_PREV 10000000

static void initArrays(long long *distanceArray, unsigned long* prevArray, unsigned long size){
    for(unsigned long i = 0; i < size; i++){
        distanceArray[i] = INFINIT_DISTANCE;
        prevArray[i] = NO_PREV;
    }
}

void bellmanFord(Graph *graph, unsigned long startVertex, long long *distanceArray, unsigned long *prevArray){
    if(!graph || !graph->edgeList || !distanceArray || !prevArray){
        return;
    }
    initArrays(distanceArray,prevArray,graph->numberOfVertices);
    distanceArray[startVertex] = 0;

    for(unsigned int n = 0; n < graph->numberOfVertices; n++){
        for(unsigned int edge = 0; edge < graph->edgeListSize; edge += 3){
            unsigned long vertex1 = (unsigned long) graph->edgeList[edge];
            unsigned long vertex2 = (unsigned long) graph->edgeList[edge+1];
            long long weight = graph->edgeList[edge+2];
            if(distanceArray[vertex1] + weight < distanceArray[vertex2]){
                distanceArray[vertex2] = distanceArray[vertex1] + weight;
                prevArray[vertex2] = vertex1;
            }
        }
    }

    // TODO: Check for negative loop.
}
