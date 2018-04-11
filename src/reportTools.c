#include "reportTools.h"

static bool cmpGraphDistMatrix(CompleteGraph *graph1, CompleteGraph *graph2) {
    if (!graph1 || !graph2) {
        return false;
    }

    if(graph1->size != graph2->size){
        printf("Compare error 1...\n");
        return false;
    }
    unsigned int y;

    for (y = 0; y < graph1->size; y++) {
        if (graph1->dist[y] != graph2->dist[y]) {
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

    srand48(10);
    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
            graph.adjMatrix[y][x] = (float) drand48();
            if(y == 0 && x == 0){
            }
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
                time = bellmanFordParallelCpu(&graphParallel, 0, currentThreadNumber);
                bool check = cmpGraphDistMatrix(&graphParallel, &graphSequ);
                printf("parallelCpu;case=%d;n=%d;time=%lf;threads=%d;check=%d\n", currentRunNumber, currentVertNumber,
                       time, currentThreadNumber, check);
                destroyCompleteGraph(&graphParallel);
            }

            destroyCompleteGraph(&graphSequ);
        }

    }

}