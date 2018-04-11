#include "reportTools.h"

#define DEBUG_MODE false

static bool cmpGraphDistMatrix(CompleteGraph *graph, float *otherDistMatrix) {
    if (!graph || !otherDistMatrix) {
        return false;
    }
    unsigned int y;
    for (y = 0; y < graph->size; y++) {
        if (graph->dist[y] != otherDistMatrix[y]) {
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
    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
            graph.adjMatrix[y][x] = (float) drand48();
        }
    }

    return graph;
}

void createReport(Report *report) {
    if (!report) {
        return;
    }
    unsigned int currentRunNumber, verticesPtr, threadPtr, currentThreadNumber, currentVertNumber;
    double time;
    for (currentRunNumber = 1; currentRunNumber <= report->numberOfRuns; currentRunNumber++) {
        for (verticesPtr = 0; verticesPtr < report->verticesCasesSize; verticesPtr++) {
            currentVertNumber = report->verticesCases[verticesPtr];

            CompleteGraph graphSequ = buildRandomCompleteGraph(currentVertNumber);
            if (graphSequ.error) {
                printf("FATAL ERROR occurred...\n");
                return;
            }
            time = bellmanFord(&graphSequ, 0);
            printf("sequ;case=%d;n=%d;time=%lf;\n", currentRunNumber, currentVertNumber, time);

            for (threadPtr = 0; threadPtr < report->threadCasesSize; threadPtr++) {
                currentThreadNumber = report->threadCases[threadPtr];
                CompleteGraph graphParallel = buildRandomCompleteGraph(currentVertNumber);
                if (graphParallel.error) {
                    printf("FATAL ERROR occurred...\n");
                    return;
                }
                bellmanFordParallelCpu(&graphParallel, 0, currentThreadNumber);
                bool check = cmpGraphDistMatrix(&graphParallel, graphSequ.dist);
                printf("parallelCpu;case=%d;n=%d;time=%lf;threads=%d;check=%d\n", currentRunNumber, currentVertNumber, time,
                       currentThreadNumber, check);
                destroyCompleteGraph(&graphParallel);
            }

            destroyCompleteGraph(&graphSequ);
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