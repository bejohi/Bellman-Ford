#include "bellmanFordCompleteGraphGpuParallel.h"
#include "bellmanFordCompleteGraphSequential.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000
#define DEBUG 1

static inline void initArrays(float *distanceArray, long size) {
    for (unsigned long i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
    }
}

static void fillGpuGraphRandom(GpuGraph *graph) {
    if (!graph) {
        return;
    }
    srand48(10);
    for (unsigned long i = 0; i < graph->size * graph->size; i++) {
        graph->adjMatrix1D[i] = drand48();
    }
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

static bool cmpDistArr(float* dist1, float* dist2, unsigned int size){
    if(!dist1 || !dist2){
        return false;
    }

    for(int i = 0; i < size; i++){
        if(dist1[i] != dist2[i]){
            return false;
        }
    }

    return true;
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

double bellmanFordGpu(GpuGraph *graph, unsigned int startVertex, unsigned int blockSize, unsigned int threadNum) {

    // CPU Setup
    if (!graph || !graph->adjMatrix1D || !graph->dist) {
        return -1;
    }
    if(DEBUG) printf("Init arrays...\n");
    initArrays(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    int *finished = (int*) malloc(sizeof(int));
    int *finishedGpu;
    unsigned int n, y, x, i;
    float *gpuadjMatrix1D;
    float *gpuDistArray;

    // GPU Setup
    if(DEBUG) printf("CUDA malloc...\n");
    CHECK(cudaMalloc((void **) &gpuadjMatrix1D, sizeof(float) * graph->size * graph->size));
    CHECK(cudaMalloc((void **) &gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((void **) &finishedGpu, sizeof(int)));
    if(DEBUG) printf("CUDA malloc done...\n");
    int grid = (graph->size * graph->size) / threadNum;

    double time = seconds();
    for (n = 0; n < graph->size; n++) {
        *finished = 1;
        if(DEBUG) printf("CUDA memcpy for n=%d...\n",n);
        CHECK(cudaMemcpy(gpuadjMatrix1D, graph->adjMatrix1D, sizeof(float) * graph->size * graph->size,
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpuDistArray, graph->dist, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(finishedGpu, finished, sizeof(int), cudaMemcpyHostToDevice));

        if(DEBUG) printf("Inner Bellmanford...\n");
        innerBellmanFord <<<grid, blockSize>>> (gpuadjMatrix1D, gpuDistArray, graph->size, finishedGpu);
        CHECK(cudaDeviceSynchronize());

        if(DEBUG) printf("CUDA memcpy back...\n");
        CHECK(cudaMemcpy(graph->adjMatrix1D, gpuadjMatrix1D, sizeof(float) * graph->size * graph->size,
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(graph->dist, gpuDistArray, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(finished, finishedGpu, sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaGetLastError());

        if (*finished) {
            break;
        }
    }
    if(DEBUG) printf("Done...\n");
    time = seconds() - time;

    CHECK(cudaFree(gpuadjMatrix1D));
    CHECK(cudaFree(gpuDistArray));
    CHECK(cudaFree(finishedGpu));

    CHECK(cudaDeviceReset());

    return time;
}

int main() {
    if(DEBUG) printf("Starting GPU Test...\n");

    // init locals
    int dev = 0;
    unsigned int n = 10000;
    unsigned int blockSize, threadsPerBlock;
    if(DEBUG) printf("Create graph...\n");
    GpuGraph graph = createGpuGraph(n);

    if(DEBUG) printf("Fill graph...\n");
    fillGpuGraphRandom(&graph);
    if(DEBUG) printf("Fill done...\n");
    CHECK(cudaSetDevice(dev));
    blockSize = 512;
    threadsPerBlock = 512;
    if(DEBUG) printf("Run gpu bellman ford...\n");
    double time = bellmanFordGpu(&graph, 0, blockSize, threadsPerBlock);
    printf("result=%lf\n",time);

    CompleteGraph cpuGraph = buildRandomCompleteGraph(n);
    bellmanFord(&cpuGraph,0);
    bool check = cmpDistArr(cpuGraph.dist,graph.dist,graph.size);
    printf("check=%d\n",check);
    

}