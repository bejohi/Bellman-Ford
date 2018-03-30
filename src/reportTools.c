#include "reportTools.h"

#define setRandomSeed() (srand((unsigned)time(NULL)))
#define randomFloat() ((float)rand()/RAND_MAX)

static CompleteGraph buildRandomCompleteGraph(unsigned int size){
    CompleteGraph graph = createCompleteGraph(size);
    if(graph.error){
        return graph;
    }

    setRandomSeed();
    for(unsigned int y = 0; y < size; y++){
        for(unsigned int x = 0; x < size; x++){
            graph.adjMatrix[y][x] = randomFloat();
        }
    }

    return graph;
}


void printReportBellmanFordCompleteGraphSequential(unsigned int* graphSizeArray, unsigned int arrSize){
    if(!graphSizeArray){
        return;
    }
    for(unsigned int i = 0; i < arrSize; i++){
        printf("Creating random graph with number of edges = %d\n",graphSizeArray[i] * graphSizeArray[i]);
        CompleteGraph graph = buildRandomCompleteGraph(graphSizeArray[i]);
        if(graph.error){
            printf("ERROR: graph could not be build with vertex number %d\n",graphSizeArray[i]);
            return;
        }
        printf("Calculating...\n");
        double bellmanFordTime = bellmanFord(&graph,0);
        printf("sequential;numberOfVertices=%d;duration=%lf\n",graphSizeArray[i],bellmanFordTime);
        destroyCompleteGraph(&graph);
    }
}