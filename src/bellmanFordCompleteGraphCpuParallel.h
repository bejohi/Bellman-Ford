#ifndef INF236_CA2_BELLMANFORDCOMPLETEGRAPHCPUPARALLEL_H
#define INF236_CA2_BELLMANFORDCOMPLETEGRAPHCPUPARALLEL_H

#include "completeGraph.h"
#include <omp.h>

double bellmanFordParallelCpu(CompleteGraph* graph, unsigned int startVertex, unsigned int numberOfThreads);

#endif //INF236_CA2_BELLMANFORDCOMPLETEGRAPHCPUPARALLEL_H
