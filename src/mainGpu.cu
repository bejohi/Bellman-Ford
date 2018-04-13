#include "mainGpu.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000
#define DEBUG 0
#define DEBUG_DEEP 0

static inline void cpuInitArr(float *distanceArray, long size) {
    unsigned long i;
    for (i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
    }
}

void cpuDestroyGraph(CpuGraph *CpuGraph) {
    free(CpuGraph->dist);
    unsigned int i;
    for (i = 0; i < CpuGraph->size; i++) {
        if (CpuGraph->adjMatrix[i]) {
            free(CpuGraph->adjMatrix[i]);
        }
    }
    free(CpuGraph->adjMatrix);
}

CpuGraph cpuCreateGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    CpuGraph CpuGraph = {.size = size, .isDirected = false};

    CpuGraph.dist = (float *) malloc(sizeof(float) * size);
    CpuGraph.adjMatrix = (float **) malloc(sizeof(float *) * size);

    if (!CpuGraph.dist || !CpuGraph.adjMatrix) {
        cpuDestroyGraph(&CpuGraph);
        return {};
    }

    unsigned int i, x;

    for (i = 0; i < size; i++) {
        CpuGraph.adjMatrix[i] = (float *) malloc(sizeof(float) * size);
        if (!CpuGraph.adjMatrix[i]) {
            cpuDestroyGraph(&CpuGraph);
            return {};
        }
        if (i == 0) {
            for (x = 0; x < size; x++) {
                CpuGraph.adjMatrix[i][x] = 0;
            }
        } else {
            memcpy(CpuGraph.adjMatrix[i], CpuGraph.adjMatrix[0], sizeof(float) * size);
        }

    }
    return CpuGraph;
}

double cpuBellmanFord(CpuGraph *graph, unsigned int startVertex) {
    if (!graph || !graph->adjMatrix || !graph->dist) {
        return -1;
    }
    cpuInitArr(graph->dist, graph->size);
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

static inline void gpuInitArrays(float *distanceArray, long size) {
    for (unsigned long i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
    }
}

static void gpuFillGraphRandom(GpuGraph *graph) {
    if (!graph) {
        return;
    }
    srand48(10);
    for (unsigned long i = 0; i < graph->size * graph->size; i++) {
        graph->adjMatrix1D[i] = drand48();
    }
}

static CpuGraph gpuBuildRandomGraph(unsigned int size) {
    CpuGraph graph = cpuCreateGraph(size);
    if (graph.error) {
        return graph;
    }

    unsigned int y, x;

    srand48(10);
    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
            graph.adjMatrix[y][x] = (float) drand48();
        }
    }

    return graph;
}

static bool cmpCpuWithGpuResult(CpuGraph *CpuGraph, GpuGraph *gpuGraph, unsigned int size) {
    if (!gpuGraph->dist || !CpuGraph->dist) {
        if (DEBUG) printf("Diff error 1\n");
        return false;
    }
    int i, y;
    if (DEBUG) {
        for (i = 0; i < size; i++) {
            for (y = 0; y < size; y++) {
                if (CpuGraph->adjMatrix[i][y] != gpuGraph->adjMatrix1D[y + (i * size)]) {
                    if (DEBUG) printf("Diff error 2 for i=%d & y=%d\n", i, y);
                    return false;
                }
            }
        }
    }

    if (DEBUG_DEEP) {
        for (i = 0; i < size; i++) {
            printf("i=%d;GPU:%lf;CPU:%lf\n", i, gpuGraph->dist[i], CpuGraph->dist[i]);
        }
    }

    for (i = 0; i < size; i++) {
        if (gpuGraph->dist[i] != CpuGraph->dist[i]) {
            if (DEBUG) printf("Diff error 3 for i=%d\n", i);
            if (DEBUG) printf("GPU: %lf vs CPU:%lf\n", gpuGraph->dist[i], CpuGraph->dist[i]);
            return false;
        }
    }


    return true;
}

GpuGraph gpuCreateGraph(unsigned int size) {
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

void gpuDestroyGraph(GpuGraph *GpuGraph) {
    free(GpuGraph->dist);
    free(GpuGraph->adjMatrix1D);
}

__global__ void innerBellmanFord(float *adjMatrix1D, float *dist, unsigned int size, int *finished) {
    unsigned int x, y, currentMatrixPosition;
    float weight;
    currentMatrixPosition = threadIdx.x + blockIdx.x * blockDim.x;

    while (currentMatrixPosition < size * size) {
        y = currentMatrixPosition / size;
        x = currentMatrixPosition % size;
        weight = adjMatrix1D[currentMatrixPosition];
        if (dist[y] + weight < dist[x]) {
            dist[x] = dist[y] + weight;
            *finished = 0;

        }
        currentMatrixPosition += gridDim.x * blockDim.x;
    }

}

double bellmanFordGpu(GpuGraph *graph, unsigned int startVertex, unsigned int blockSize, unsigned int threadNum) {

    // PRECHECK
    if (!graph || !graph->adjMatrix1D || !graph->dist) {
        return -1;
    }

    // INIT LOCALS
    if (DEBUG) printf("Init arrays...\n");
    gpuInitArrays(graph->dist, graph->size);
    graph->dist[startVertex] = 0;
    int *finished = (int *) malloc(sizeof(int));
    int *finishedGpu;
    unsigned int n;
    float *gpuadjMatrix1D;
    float *gpuDistArray;
    unsigned long size2D = sizeof(float) * graph->size * graph->size;

    // GPU SETUP
    if (DEBUG) printf("CUDA malloc...\n");
    CHECK(cudaMalloc((void **) &gpuadjMatrix1D, size2D));
    CHECK(cudaMalloc((void **) &gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((void **) &finishedGpu, sizeof(int)));
    CHECK(cudaMemcpy(gpuDistArray, graph->dist, sizeof(float) * graph->size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpuadjMatrix1D, graph->adjMatrix1D, size2D, cudaMemcpyHostToDevice));
    if (DEBUG) printf("CUDA malloc done...\n");
    dim3 block(blockSize);
    dim3 grid(threadNum);

    // START INNER LOOP
    double time = seconds();
    for (n = 0; n < graph->size; n++) {
        *finished = 1;
        if (DEBUG) printf("CUDA memcpy for n=%d...\n", n);


        CHECK(cudaMemcpy(finishedGpu, finished, sizeof(int), cudaMemcpyHostToDevice));

        innerBellmanFord <<<grid, block>>> (gpuadjMatrix1D, gpuDistArray, graph->size, finishedGpu);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(finished, finishedGpu, sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaGetLastError());

        if (*finished) {
            if (DEBUG) printf("True Finished with n=%d...\n", n);
            break;
        }
    }
    time = seconds() - time;

    // CLEAN-UP
    CHECK(cudaMemcpy(graph->dist, gpuDistArray, sizeof(float) * graph->size, cudaMemcpyDeviceToHost));
    if (DEBUG) printf("Done...\n");
    CHECK(cudaFree(gpuadjMatrix1D));
    CHECK(cudaFree(gpuDistArray));
    CHECK(cudaFree(finishedGpu));

    CHECK(cudaDeviceReset());

    free(finished);

    return time;
}

static void createReport() {
    printf("# Create report...\n");
    unsigned int n = 10000;
    GpuGraph gpuGraph;
    unsigned int threadArr[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    unsigned int blockArr[] = {50,100,150,200,250,300,350,400,500,1000,1024,2000,3000,4000,5000,8000,10000};
    printf("# Pre Build...\n");
    CpuGraph cpuGrap = gpuBuildRandomGraph(n);
    cpuBellmanFord(&cpuGrap, 0);
    printf("# ...done! Run test cases...\n");
    for (unsigned int tPtr = 0; tPtr < 11; tPtr++) {
        for (unsigned int bPtr = 0; bPtr < 17; bPtr++) {
            gpuGraph = gpuCreateGraph(n);
            gpuFillGraphRandom(&gpuGraph);
            if(DEBUG) printf("Run with thread=%d & block=%d\n",threadArr[tPtr],blockArr[bPtr]);
            double time = bellmanFordGpu(&gpuGraph, 0, blockArr[bPtr], threadArr[tPtr]);
            bool check = cmpCpuWithGpuResult(&cpuGrap, &gpuGraph,gpuGraph.size);
            printf("parallelGpu;n=%d;threads=%d;blockSize=%d;time=%lf;check=%d;\n", n, threadArr[tPtr],
                   blockArr[bPtr], time, check);
            gpuDestroyGraph(&gpuGraph);
        }
    }
    cpuDestroyGraph(&cpuGrap);


}

static void preTest() {
    printf("Starting GPU Test...\n");

    // init locals
    int dev = 0;
    unsigned int n = 10000;
    unsigned int blockSize, threadsPerBlock;
    if (DEBUG) printf("Create graph...\n");
    GpuGraph graph = gpuCreateGraph(n);

    if (DEBUG) printf("Fill graph...\n");
    gpuFillGraphRandom(&graph);
    if (DEBUG) printf("Fill done...\n");
    CHECK(cudaSetDevice(dev));
    blockSize = 512;
    threadsPerBlock = n;
    if (DEBUG) printf("Run gpu bellman ford...\n");
    double time = bellmanFordGpu(&graph, 0, blockSize, threadsPerBlock);
    printf("result=%lf\n", time);
    if (DEBUG) printf("Build cpu graph...\n");
    CpuGraph cpuGraph = gpuBuildRandomGraph(n);
    if (DEBUG) printf("Run cpu bellman-ford...\n");
    cpuBellmanFord(&cpuGraph, 0);
    if (DEBUG) printf("Run check...\n");
    bool check = cmpCpuWithGpuResult(&cpuGraph, &graph, graph.size);
    printf("check=%d\n", check);
    gpuDestroyGraph(&graph);
    cpuDestroyGraph(&cpuGraph);
}

int main() {
    if(DEBUG) preTest();
    createReport();
}