#include "testCompleteGraph.h"

bool test_createCompleteGraph() {
    // Arrange
    unsigned int size = 10000;

    // Act
    CompleteGraph graph = createCompleteGraph(size);

    // Assert
    if (graph.size != size || !graph.dist || !graph.predecessor || !graph.adjMatrix) {
        return false;
    }

    if (!graph.adjMatrix[0] || graph.adjMatrix[0][0] != 0 ||
        !graph.adjMatrix[size - 1] || graph.adjMatrix[size - 1][size - 1] != 0) {
        return false;
    }

    destroyCompleteGraph(&graph);
    return true;
}