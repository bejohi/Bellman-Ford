#include <stdbool.h>
#include "task1.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 10000000
#define NO_PREV 10000000

static void initArrays(float *distanceArray, long* prevArray, long size){
    for(unsigned long i = 0; i < size; i++){
        distanceArray[i] = INFINIT_DISTANCE;
        prevArray[i] = NO_PREV;
    }
}

void bellmanFord(Graph *graph, long startVertex, float *distanceArray, long *prevArray){
    if(!graph || !graph->edgeList || !distanceArray || !prevArray){
        return;
    }
    initArrays(distanceArray,prevArray,graph->numberOfVertices);
    distanceArray[startVertex] = 0;

    bool finished;
    for(unsigned int n = 0; n < graph->numberOfVertices; n++){
        finished = true;
        for(unsigned int edge = 0; edge < graph->edgeListSize; edge += 3){
            long vertex1 = (long) graph->edgeList[edge];
            long vertex2 = (long) graph->edgeList[edge+1];
            float weight = graph->edgeList[edge+2];
            if(distanceArray[vertex1] + weight < distanceArray[vertex2]){
                finished = false;
                distanceArray[vertex2] = distanceArray[vertex1] + weight;
                prevArray[vertex2] = vertex1;
            }
        }
        if(finished){
            break;
        }
    }

    // TODO: Check for negative loop.
}
