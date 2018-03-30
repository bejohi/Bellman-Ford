#ifndef INF236_CA2_TASK1_H
#define INF236_CA2_TASK1_H

#include "completeGraph.h"
#include <omp.h>

/**
 * Runs the Bellman-Ford Algorithm on a given Complete Graph.
 * The distance and predecessor arrays of the graph will be filled with the matching values.
 * @param graph the CompleteGraph struct. Must be initialized
 * @param startVertex the index of the start Vertex.
 * @return the time it took to run the algorithm in seconds.
 */
double bellmanFord(CompleteGraph *graph, unsigned int startVertex);

#endif //INF236_CA2_TASK1_H
