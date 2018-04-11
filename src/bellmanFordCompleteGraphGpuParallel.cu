#include "bellmanFordCompleteGraphGpuParallel.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000

inline void initArrays(float *distanceArray, long size) {
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

    unsigned int i;

    for (i = 0; i < size * size; i++) {
        GpuGraph.adjMatrix1D[i] = 0;
    }
    return GpuGraph;
}


void destroyGpuGraph(GpuGraph *GpuGraph) {
    free(GpuGraph->dist);
    free(GpuGraph->adjMatrix1D);
}


__global__ void innerBellmanFord(float *adjMatrix1D, float *dist, unsigned int size, int *finished) {
    unsigned int x, y, currentMatrixPosition;
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
    } while (currentMatrixPosition < size * size);

}

double bellmanFordGpu(GpuGraph *graph, unsigned int startVertex, unsigned int blockSize, unsigned int threadsPerBlock) {

    // CPU Setup
    if (!graph || !graph->adjMatrix1D || !graph->dist) {
        return -1;
    }

    initArrays(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    int *finished;
    int *finishedGpu;
    unsigned int n, y, x, i;
    float *gpuadjMatrix1D;
    float *gpuDistArray;

    // GPU Setup
    CHECK(cudaMalloc((float **) gpuadjMatrix1D, sizeof(float) * graph->size * graph->size));
    CHECK(cudaMalloc((float **) gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((int **) finishedGpu, sizeof(bool)));

    int dimx = 32;
    dim3 block;
    dim3 grid;


    double time = seconds();
    for (n = 0; n < graph->size; n++) {
        *finished = 1;
        CHECK(cudaMemcpy(gpuadjMatrix1D, graph->adjMatrix1D, sizeof(float) * graph->size * graph->size,
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpuDistArray, graph->dist, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(finishedGpu, finished, sizeof(int), cudaMemcpyHostToDevice));

        innerBellmanFord <<<grid, block>>>(gpuadjMatrix1D, gpuDistArray, graph->size, finishedGpu);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(graph->adjMatrix1D, gpuadjMatrix1D, sizeof(float) * graph->size * graph->size,
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(graph->dist, gpuDistArray, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(finished, finishedGpu, sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaGetLastError());

        if (*finished) {
            break;
        }
    }
    time = seconds() - time;
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