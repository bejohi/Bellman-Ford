#include "reportTools.h"

#define DEBUG_MODE false
#define setRandomSeed() (srand((unsigned)time(NULL)))
#define randomFloat() ((float)rand()/RAND_MAX)

static CompleteGraph buildRandomCompleteGraph(unsigned int size) {
    CompleteGraph graph = createCompleteGraph(size);
    if (graph.error) {
        return graph;
    }

    setRandomSeed();
    for (unsigned int y = 0; y < size; y++) {
        for (unsigned int x = 0; x < size; x++) {
            graph.adjMatrix[y][x] = randomFloat();
        }
    }

    return graph;
}

void createReportParallelCpu(Report *report) {
    if (!report) {
        return;
    }
    for (unsigned int threadPtr = 0; threadPtr < report->threadCasesSize; threadPtr++) {
        for (unsigned int verticesPtr = 0; verticesPtr < report->verticesCasesSize; verticesPtr++) {
            unsigned int numberOfVertices = report->verticesCases[verticesPtr];
            CompleteGraph graph = buildRandomCompleteGraph(numberOfVertices);
            if (graph.error) {
                printf("ERROR: graph could not be build with vertex number %d\n", numberOfVertices);
                return;
            }
            double bellmanFordTime = bellmanFordParallelCpu(&graph, 0,report->threadCases[threadPtr]);
            printf("parallelCpu;numberOfVertices=%d;numberOfEdges=%d,threads=%d,duration=%lf\n", numberOfVertices,
                   numberOfVertices * numberOfVertices, report->threadCases[threadPtr],bellmanFordTime);
            destroyCompleteGraph(&graph);
        }
    }
}

void printReportBellmanFordCompleteGraphSequential(unsigned int *graphSizeArray, unsigned int arrSize) {
    if (!graphSizeArray) {
        return;
    }
    for (unsigned int i = 0; i < arrSize; i++) {
        if (DEBUG_MODE)
            printf("Creating random graph with number of edges = %d\n", graphSizeArray[i] * graphSizeArray[i]);
        CompleteGraph graph = buildRandomCompleteGraph(graphSizeArray[i]);
        if (graph.error) {
            printf("ERROR: graph could not be build with vertex number %d\n", graphSizeArray[i]);
            return;
        }
        if (DEBUG_MODE) printf("Calculating...\n");
        double bellmanFordTime = bellmanFord(&graph, 0);
        printf("sequential;numberOfVertices=%d;numberOfEdges=%d,duration=%lf\n", graphSizeArray[i],
               graphSizeArray[i] * graphSizeArray[i], bellmanFordTime);
        destroyCompleteGraph(&graph);
    }
}