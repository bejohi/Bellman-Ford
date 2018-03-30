#include "testBellmanForcCompleteGraphSequential.h"

bool test_bellmanFord() {
    // Arrange
    CompleteGraph completeGraph = createCompleteGraph(3);
    addEdgeCompleteGraph(&completeGraph, 0, 1, 5);
    addEdgeCompleteGraph(&completeGraph, 1, 0, 3);

    addEdgeCompleteGraph(&completeGraph, 0, 2, 10);
    addEdgeCompleteGraph(&completeGraph, 2, 0, 8);

    addEdgeCompleteGraph(&completeGraph, 1, 2, 99);
    addEdgeCompleteGraph(&completeGraph, 2, 1, 5);

    // Act
    bellmanFord(&completeGraph, 0);

    // Assert
    if (completeGraph.dist[0] != 0 ||
        completeGraph.dist[1] != 5 ||
        completeGraph.dist[2] != 10) {
        return false;
    }

    destroyCompleteGraph(&completeGraph);
    return true;

}