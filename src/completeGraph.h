#ifndef INF236_CA2_COMPLETEGRAPH_H
#define INF236_CA2_COMPLETEGRAPH_H

#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>

#define MAX_GRAPH_SIZE 10000

/**
 * A representation of a complete graph. Can be directed, or undirected. Stores edges in an adjacency matrix.
 */
typedef struct CompleteGraph {
    unsigned int size; //< the number of vertices.
    bool isDirected; //< indicates if the graph is directed.
    bool error; //< a flag which will be true if any function call on the graph struct causes an error.
    float **adjMatrix; //< a 2D matrix with the dimensions of size * size, where every colume indicates the distance between 2 vertices.
    float *dist; //< Stores the distance to a start vertex. Can be filled with shortest path algorithm.
    unsigned int *predecessor; //< Stores the predecessor for all vertices. Can be filled with shortest path algorithm.
} CompleteGraph;


/**
 * Frees all used memory used by a graph struct.
 * @param completeGraph a pointer to the graph to destroy.
 */
void destroyCompleteGraph(CompleteGraph *completeGraph);

/**
 * Creates a complete undirected graph with the given size. Init all arrays inside the graph struct.
 * @param size the number of vertices.
 * @return the newly created Graph Struct.
 */
CompleteGraph createCompleteGraph(unsigned int size);

/**
 * Adds an edge to the graph.
 * @param graph the graph.
 * @param startVertex the index of the start vertex
 * @param endVertex the index of the end vertex.
 * @param weight the weight of the edge.
 */
void addEdgeCompleteGraph(CompleteGraph *graph, unsigned int startVertex, unsigned int endVertex, float weight);

void initArrays(float *distanceArray, unsigned int* prevArray, long size);

#endif //INF236_CA2_COMPLETEGRAPH_H
