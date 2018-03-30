#include "task1Test.h"

bool testBellmanFord() {
    // Arrange
    Graph graph = createGraph(10, 10);
    addEdge(&graph, 0, 1, 10);
    addEdge(&graph, 1, 2, 5);
    addEdge(&graph, 0, 3, 5);
    addEdge(&graph, 3, 2, 10);
    addEdge(&graph, 2, 4, 5);
    float *distanceArray = (float *) malloc(sizeof(float) * graph.numberOfVertices);
    long *prevArray = (long *) malloc(sizeof(long) * graph.numberOfVertices);

    // Act
    bellmanFord(&graph, 0, distanceArray, prevArray);

    // Assert
    if (distanceArray[0] != 0 ||
        distanceArray[1] != 10 ||
        distanceArray[2] != 15 ||
        distanceArray[3] != 5 ||
        distanceArray[4] != 20) {
        return false;
    }

    destroyGraph(&graph);
    free(prevArray);
    free(distanceArray);
    return true;
}