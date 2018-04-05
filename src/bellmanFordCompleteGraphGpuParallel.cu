#include "bellmanFordCompleteGraphGpuParallel.h"

// TODO: Use better values.
#define INFINIT_DISTANCE 1000000
#define NO_PREV 100000

inline void initArrays(float *distanceArray, unsigned int *prevArray, long size) {
    for (unsigned long i = 0; i < size; i++) {
        distanceArray[i] = INFINIT_DISTANCE;
        prevArray[i] = NO_PREV;
    }
}

CompleteGraph createCompleteGraph(unsigned int size) {
    if (size > MAX_GRAPH_SIZE) {
        size = MAX_GRAPH_SIZE;
    }
    CompleteGraph completeGraph = {.size = size, .isDirected = false, .error = false};

    completeGraph.dist = (float *) malloc(sizeof(float) * size);
    completeGraph.adjMatrix = (float *) malloc(sizeof(float) * size * size);

    if (!completeGraph.dist || !completeGraph.adjMatrix) {
        exit(-101);
    }

    unsigned int i, x;

    for (i = 0; i < size * size; i++) {
        completeGraph.adjMatrix[i] = 0;
    }
    return completeGraph;
}

void addEdgeCompleteGraph(CompleteGraph *graph, unsigned int startVertex, unsigned int endVertex, float weight) {
    if (!graph) {
        exit(-103);
    }
    if (!graph->adjMatrix || endVertex >= graph->size || startVertex >= graph->size) {
        exit(-102);
    }

    graph->adjMatrix[startVertex + ] = weight;
    if (graph->isDirected) {
        graph->adjMatrix[endVertex][startVertex] = weight;
    }

}


void destroyCompleteGraph(CompleteGraph *completeGraph) {
    free(completeGraph->predecessor);
    free(completeGraph->dist);
    unsigned int i;
    for (i = 0; i < completeGraph->size; i++) {
        if (completeGraph->adjMatrix[i]) {
            free(completeGraph->adjMatrix[i]);
        }
    }
    free(completeGraph->adjMatrix);
}


__global__ innerBellmanFord(float *adjMatrix, float *dist, unsigned int size, int* finished) {
    unsigned int x,y,currentMatrixPosition;
    currentMatrixPosition = threadIdx.x + blockIdx.x * blockDim.x;
    do {
        y = currentMatrixPosition / size;
        x = currentMatrixPosition & size;
        float weight = adjMatrix[currentMatrxiPosition];
        if (dist[y] + weight < dist[x]) {
            dist[x] = dist[y] + weight;
            finished = 0;

        }
        currentMatrixPosition += gridDim.x * blockDim.x;
    } while(currentMatrixPosition < size * size);

}

double bellmanFordGpu(CompleteGraph *graph, unsigned int startVertex) {

    // CPU Setup
    if (!graph || !graph->adjMatrix || !graph->predecessor || !graph->dist) {
        return -1;
    }

    initArrays(graph->dist, graph->predecessor, graph->size);
    graph->dist[startVertex] = 0;
    double starttime, endtime;
    bool finished;
    bool* finishedGpu;
    unsigned int n, y, x, i;
    float** gpuAdjMatrix;
    float* gpuDistArray;

    // GPU Setup
    CHECK(cudaMalloc((float*) gpuAdjMatrix, sizeof(float) * graph->size * graph->size));
    CHECK(cudaMalloc((float*) gpuDistArray, sizeof(float) * graph->size));
    CHECK(cudaMalloc((bool*) finishedGpu, sizeof(bool)));

    // TODO: Init Arrays for GPU

    for (n = 0; n < graph->size; n++) {
        finished = true;

        //innerBellmanFord()
        for (y = 0; y < graph->size; y++) {
            for (x = 0; x < graph->size; x++) {
                float weight = graph->adjMatrix[y][x];
                if (graph->dist[y] + weight < graph->dist[x]) {
                    graph->dist[x] = graph->dist[y] + weight;
                    graph->predecessor[x] = y;
                    finished = false;
                }
            }
        }
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