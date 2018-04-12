#include "bellmanFordCompleteGraphGpuParallel.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000
#define DEBUG 1


// REGION: SEQU Graph
typedef struct CompleteGraph {
    unsigned int size; //< the number of vertices.
    bool isDirected; //< indicates if the graph is directed.
    bool error; //< a flag which will be true if any function call on the graph struct causes an error.
    float **adjMatrix; //< a 2D matrix with the dimensions of size * size, where every colume indicates the distance between 2 vertices.
    float *dist; //< Stores the distance to a start vertex. Can be filled with shortest path algorithm.
} CompleteGraph;

inline void initArraysSequ(float *distanceArray,long size) {
    unsigned long i;
    for (i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
    }
}

void destroyCompleteGraph(CompleteGraph *completeGraph) {
    free(completeGraph->dist);
    unsigned int i;
    for (i = 0; i < completeGraph->size; i++) {
        if (completeGraph->adjMatrix[i]) {
            free(completeGraph->adjMatrix[i]);
        }
    }
    free(completeGraph->adjMatrix);
}

CompleteGraph createCompleteGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    CompleteGraph completeGraph = {.size = size, .isDirected = false};

    completeGraph.dist = (float *) malloc(sizeof(float) * size);
    completeGraph.adjMatrix = (float **) malloc(sizeof(float *) * size);

    if (!completeGraph.dist || !completeGraph.adjMatrix) {
        destroyCompleteGraph(&completeGraph);
        return {};
    }

    unsigned int i, x;

    for (i = 0; i < size; i++) {
        completeGraph.adjMatrix[i] = (float *) malloc(sizeof(float) * size);
        if (!completeGraph.adjMatrix[i]) {
            destroyCompleteGraph(&completeGraph);
            return {};
        }
        if (i == 0) {
            for (x = 0; x < size; x++) {
                completeGraph.adjMatrix[i][x] = 0;
            }
        } else {
            memcpy(completeGraph.adjMatrix[i], completeGraph.adjMatrix[0], sizeof(float) * size);
        }

    }
    return completeGraph;
}

double bellmanFord(CompleteGraph *graph, unsigned int startVertex) {
    if (!graph || !graph->adjMatrix || !graph->dist) {
        return -1;
    }
    initArraysSequ(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    double startTime, endTime;
    bool finished;
    unsigned int n, y, x;
    startTime = seconds();
    for (n = 0; n < graph->size; n++) {
        finished = true;
        for (y = 0; y < graph->size; y++) {
            for (x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                if (graph->dist[y] + weight < graph->dist[x]) {
                    graph->dist[x] = graph->dist[y] + weight;
                    finished = false;
                }
            }
        }
        if (finished) {
            break;
        }
    }
    endTime = seconds();
    return endTime - startTime;
}

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

static bool cmpDistArr(CompleteGraph* completeGraph, GpuGraph* gpuGraph, unsigned int size){
    if(!gpuGraph->dist || !completeGraph->dist){
        if(DEBUG) printf("Diff error 1\n");
        return false;
    }
    int i,y;
    if(DEBUG){
        for(i = 0; i < size; i++){
            for(y = 0; y < size; y++){
                if(completeGraph->adjMatrix[i][y] != gpuGraph->adjMatrix1D[y+(i*size)]){
                    if(DEBUG) printf("Diff error 2 for i=%d & y=%d\n",i,y);
                    return false;
                }
            }
        }
    }
    for(i = 0; i < size; i++){
        if(gpuGraph->dist[i] != completeGraph->dist[i]){
            if(DEBUG) printf("Diff error 3 for i=%d\n",i);
            if(DEBUG) printf("GPU: %lf vs CPU:%lf\n",gpuGraph->dist[i], completeGraph->dist[i]);
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
        x = currentMatrixPosition / size;
        y = currentMatrixPosition & size;
        float weight = adjMatrix1D[currentMatrixPosition];
        if (dist[y] + weight < dist[x]) {
            dist[x] = dist[y] + weight;
            *finished = 0;

        }
        currentMatrixPosition += gridDim.x * blockDim.x;
    } while (currentMatrixPosition < size * size);

}

double bellmanFordGpu(GpuGraph *graph, unsigned int startVertex, unsigned int blockSize, unsigned int threadNum) {

    if (!graph || !graph->adjMatrix1D || !graph->dist) {
        return -1;
    }
    if(DEBUG) printf("Init arrays...\n");
    initArrays(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    int *finished = (int*) malloc(sizeof(int));
    int *finishedGpu;
    unsigned int n;
    float *gpuadjMatrix1D;
    float *gpuDistArray;
    unsigned long size2D = sizeof(float) * graph->size * graph->size;

    // GPU Setup
    if(DEBUG) printf("CUDA malloc...\n");
    CHECK(cudaMalloc((void **) &gpuadjMatrix1D, size2D));
    CHECK(cudaMalloc((void **) &gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((void **) &finishedGpu, sizeof(int)));
    if(DEBUG) printf("CUDA malloc done...\n");
    int grid = (graph->size * graph->size) / threadNum;

    double time = seconds();
    for (n = 0; n < graph->size; n++) {
        *finished = 1;
        if(DEBUG) printf("CUDA memcpy for n=%d...\n",n);
        CHECK(cudaMemcpy(gpuadjMatrix1D, graph->adjMatrix1D, size2D, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpuDistArray, graph->dist, sizeof(float) * graph->size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(finishedGpu, finished, sizeof(int), cudaMemcpyHostToDevice));

        if(DEBUG) printf("Inner Bellmanford...\n");
        innerBellmanFord <<<grid, blockSize>>> (gpuadjMatrix1D, gpuDistArray, graph->size, finishedGpu);
        CHECK(cudaDeviceSynchronize());

        if(DEBUG) printf("CUDA memcpy back...\n");
        CHECK(cudaMemcpy(graph->adjMatrix1D, gpuadjMatrix1D, size2D, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(graph->dist, gpuDistArray, sizeof(float) * graph->size, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(finished, finishedGpu, sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaGetLastError());

        if (*finished) {
            printf("True Finished with n=%d...\n",n);
            break;
        }
    }
    if(DEBUG) printf("Done...\n");
    time = seconds() - time;

    CHECK(cudaFree(gpuadjMatrix1D));
    CHECK(cudaFree(gpuDistArray));
    CHECK(cudaFree(finishedGpu));

    CHECK(cudaDeviceReset());

    free(finished);

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
    if(DEBUG) printf("Build cpu graph...\n");
    CompleteGraph cpuGraph = buildRandomCompleteGraph(n);
    if(DEBUG) printf("Run cpu bellman-ford...\n");
    bellmanFord(&cpuGraph,0);
    if(DEBUG) printf("Run check...\n");
    bool check = cmpDistArr(&cpuGraph,&graph,graph.size);
    printf("check=%d\n",check);
    

}