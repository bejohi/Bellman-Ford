#include "bellmanFordGpuGraphGpuParallel.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000

inline void initArrays(float *distanceArray,long size) {
    for (unsigned long i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
    }
}

GpuGraph createGpuGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    GpuGraph GpuGraph = {.size = size, .isDirected = false, .error = false};

    GpuGraph.dist = (float *) malloc(sizeof(float) * size);
    GpuGraph.adjMatrix1D = (float *) malloc(sizeof(float) * size * size);

    if (!GpuGraph.dist || !GpuGraph.adjMatrix1D) {
        exit(-101);
    }

    unsigned int i, x;

    for (i = 0; i < size * size; i++) {
        GpuGraph.adjMatrix1D[i] = 0;
    }
    return GpuGraph;
}


void destroyGpuGraph(GpuGraph *GpuGraph) {
    free(GpuGraph->dist);
    free(GpuGraph->adjMatrix1D);
}


__global__ void innerBellmanFord(float *adjMatrix1D, float *dist, unsigned int size, int* finished) {
    unsigned int x,y,currentMatrixPosition;
    currentMatrixPosition = threadIdx.x + blockIdx.x * blockDim.x;
    do {
        y = currentMatrixPosition / size;
        x = currentMatrixPosition & size;
        float weight = adjMatrix1D[currentMatrixPosition];
        if (dist[y] + weight < dist[x]) {
            dist[x] = dist[y] + weight;
            finished = 0;

        }
        currentMatrixPosition += gridDim.x * blockDim.x;
    } while(currentMatrixPosition < size * size);

}

double bellmanFordGpu(GpuGraph *graph, unsigned int startVertex) {

    // CPU Setup
    if (!graph || !graph->adjMatrix1D|| !graph->dist) {
        return -1;
    }

    initArrays(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    bool finished;
    int* finishedGpu;
    unsigned int n, y, x, i;
    float** gpuadjMatrix1D;
    float* gpuDistArray;

    // GPU Setup
    CHECK(cudaMalloc((float**) gpuadjMatrix1D, sizeof(float) * graph->size * graph->size));
    CHECK(cudaMalloc((float**) gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((int**) finishedGpu, sizeof(bool)));

    // TODO: Init Arrays for GPU

    for (n = 0; n < graph->size; n++) {
        finished = 1;

        //innerBellmanFord()
        if (finished) {
            break;
        }
    }
    return -1;
}

int main() {
    printf("Starting GPU Test...\n");

    // init locals
    int dev = 0;
    unsigned int n = 10000;
    unsigned int blockSize, threadsPerBlock;

    // GPU Setup
    CHECK(cudaSetDevice(dev));

}