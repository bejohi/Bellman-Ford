#ifndef INF236_CA2_REPORTTOOL_H
#define INF236_CA2_REPORTTOOL_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "completeGraph.h"
#include "bellmanFordCompleteGraphSequential.h"
#include "bellmanFordCompleteGraphCpuParallel.h"
#include <math.h>
#include <time.h>

#define MAX_THREAD_CASES 100
#define MAX_VERTICES_CASES 100

typedef struct Report{
    unsigned int verticesCases[MAX_VERTICES_CASES];
    unsigned int verticesCasesSize;
    unsigned int threadCases[MAX_THREAD_CASES];
    unsigned int threadCasesSize;
    char* report;
} Report;

void createReportParallelCpu(Report* report);

void printReportBellmanFordCompleteGraphSequential(unsigned int* graphSizeArray, unsigned int arrSize);

#endif //INF236_CA2_REPORTTOOL_H
