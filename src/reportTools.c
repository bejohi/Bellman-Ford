#include "reportTools.h"

#define DEBUG_MODE false
#define setRandomSeed() (srand((unsigned)time(NULL)))
#define randomFloat() ((float)rand()/RAND_MAX)


static void cpyAdjMatrix(CompleteGraph *graph, float **newAdjMatrix) {
    if (!newAdjMatrix) {
        newAdjMatrix = (float **) malloc(sizeof(float *) * graph->size);
    }
    unsigned int i, y;
    for (i = 0; i < graph->size; i++) {
        newAdjMatrix[i] = (float *) malloc(sizeof(float) * graph->size);
    }

    for (y = 0; y < graph->size; y++) {
        memcpy(newAdjMatrix[y], graph->adjMatrix[y], graph->size);
    }
}

static bool cmpGraphMatrix(CompleteGraph *graph, float **adjMatrix) {
    if (!graph || !adjMatrix) {
        return false;
    }
    unsigned int i;
    for (i = 0; i < graph->size; i++) {
        if (memcmp(graph->adjMatrix[i], adjMatrix[i], graph->size) != 0) {
            return false;
        }
    }


    return true;
}

static CompleteGraph buildRandomCompleteGraph(unsigned int size) {
    CompleteGraph graph = createCompleteGraph(size);
    if (graph.error) {
        return graph;
    }

    unsigned int y, x;
    setRandomSeed();
    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
            graph.adjMatrix[y][x] = randomFloat();
        }
    }

    return graph;
}

void createReport(Report *report) {
    if (!report) {
        return;
    }
    unsigned int runPtr, verticesPtr, threadPtr, i;
    for (runPtr = 1; runPtr <= report->numberOfRuns; runPtr++) {
        for (verticesPtr = 0; verticesPtr < report->verticesCasesSize; verticesPtr++) {
            unsigned int numberOfVertices = report->verticesCases[verticesPtr];
            unsigned int numberOfEdges = numberOfVertices * numberOfVertices;
            CompleteGraph graph = buildRandomCompleteGraph(numberOfVertices);
            float **cmpMatrix = (float **) malloc(sizeof(float *) * numberOfVertices);
            double resultTime = bellmanFord(&graph, 0);
            cpyAdjMatrix(&graph, cmpMatrix);
            printf("seq;run=%d;time=%lf;vertices=%d;edges=%d\n", runPtr, resultTime, numberOfVertices, numberOfEdges);
            for (threadPtr = 0; threadPtr < report->threadCasesSize; threadPtr++) {
                unsigned int numberOfThreads = report->threadCases[threadPtr];
                resultTime = bellmanFordParallelCpu(&graph, numberOfVertices, numberOfThreads);
                bool checkEqual = cmpGraphMatrix(&graph, cmpMatrix);
                printf("parallelCpu;run=%d;time=%lf;vertices=%d;edges=%d;threads=%d;checkEqual=%d\n", runPtr,
                       resultTime, numberOfVertices, numberOfEdges, numberOfThreads, checkEqual);

            }
            destroyCompleteGraph(&graph);
            for (i = 0; i < numberOfVertices; i++) {
                free(cmpMatrix[i]);
            }
            free(cmpMatrix);
        }

    }

}


// TODO: Validate result with sequential result.
void createReportParallelCpu(Report *report) {
    if (!report) {
        return;
    }
    unsigned int threadPtr, verticesPtr;
    for (threadPtr = 0; threadPtr < report->threadCasesSize; threadPtr++) {
        for (verticesPtr = 0; verticesPtr < report->verticesCasesSize; verticesPtr++) {
            unsigned int numberOfVertices = report->verticesCases[verticesPtr];
            CompleteGraph graph = buildRandomCompleteGraph(numberOfVertices);
            if (graph.error) {
                printf("ERROR: graph could not be build with vertex number %d\n", numberOfVertices);
                return;
            }
            double bellmanFordTime = bellmanFordParallelCpu(&graph, 0, report->threadCases[threadPtr]);
            printf("parallelCpu;numberOfVertices=%d;numberOfEdges=%d;threads=%d;duration=%lf\n", numberOfVertices,
                   numberOfVertices * numberOfVertices, report->threadCases[threadPtr], bellmanFordTime);
            destroyCompleteGraph(&graph);
        }
    }
}

void printReportBellmanFordCompleteGraphSequential(unsigned int *graphSizeArray, unsigned int arrSize) {
    if (!graphSizeArray) {
        return;
    }
    unsigned int i;
    for (i = 0; i < arrSize; i++) {
        if (DEBUG_MODE)
            printf("Creating random graph with number of edges = %d\n", graphSizeArray[i] * graphSizeArray[i]);
        CompleteGraph graph = buildRandomCompleteGraph(graphSizeArray[i]);
        if (graph.error) {
            printf("ERROR: graph could not be build with vertex number %d\n", graphSizeArray[i]);
            return;
        }
        if (DEBUG_MODE) printf("Calculating...\n");
        double bellmanFordTime = bellmanFord(&graph, 0);
        printf("sequential;numberOfVertices=%d;numberOfEdges=%d;duration=%lf\n", graphSizeArray[i],
               graphSizeArray[i] * graphSizeArray[i], bellmanFordTime);
        destroyCompleteGraph(&graph);
    }
}