#ifndef INF236_CA2_BELLMANFORDGpuGraphGPU_H
#define INF236_CA2_BELLMANFORDGpuGraphGPU_H

// #define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include <string.h>

#define MAX_GRAPH_SIZE 10000

typedef struct GpuGraph {
    unsigned int size; //< the number of vertices.
    bool isDirected; //< indicates if the graph is directed.
    bool error; //< a flag which will be true if any function call on the graph struct causes an error.
    float *adjMatrix1D; //< a 1D "matrix" with the dimensions of size * size, where every colume indicates the distance between 2 vertices.
    float *dist; //< Stores the distance to a start vertex. Can be filled with shortest path algorithm.
} GpuGraph;

// From Professional CUDA C Programming 2015

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(-101);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif //INF236_CA2_BELLMANFORDGpuGraphGPU_H
