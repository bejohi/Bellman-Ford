#ifndef INF236_CA2_REPORTTOOL_H
#define INF236_CA2_REPORTTOOL_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "completeGraph.h"
#include "bellmanFordCompleteGraphSequential.h"
#include <math.h>
#include <time.h>

void printReportBellmanFordCompleteGraphSequential(unsigned int* graphSizeArray, unsigned int arrSize);

#endif //INF236_CA2_REPORTTOOL_H
