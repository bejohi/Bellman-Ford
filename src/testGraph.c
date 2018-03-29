#include <stdio.h>
#include <stdbool.h>
#include "graph.h"
#include "testGraph.h"

static unsigned int testCounter = 0;

static bool test_createGraph_sizeIs10000_graphCorrectCreated() {
    // Arrange
    unsigned int size = 10000;
    // Act
    Graph graph = createGraph(size);

    // Assert
    if (graph.size != size) {
        return false;
    }

    if (graph.adjMatrix[0][0] != noEdge) {
        return false;
    }

    if (graph.adjMatrix[size - 1][size - 1] != noEdge) {
        return false;
    }

    destroyGraph(&graph);
    return true;
}


static bool test_addEdge_edgeIsInBounds_EdgeAdded() {
    // Arrange
    Graph graph = createGraph(10);

    // Act
    addEdge(&graph, 5, 4, 10.5);

    // Assert
    if (graph.adjMatrix[5][4] != 10.5 || graph.adjMatrix[4][5] != 10.5) {
        return false;
    }

    destroyGraph(&graph);
    return true;
}

static void printTestResult(bool result, char *testName) {
    if (!result) {
        printf("TEST FAILED: %s \n", testName);
    }
    testCounter++;
}

void runTestSuite() {
    printf("RUN Test Suite\n");

    printTestResult(test_createGraph_sizeIs10000_graphCorrectCreated(),
                    "test_createGraph_sizeIs10000_graphCorrectCreated");
    printTestResult(test_addEdge_edgeIsInBounds_EdgeAdded(), "test_addEdge_edgeIsInBounds_EdgeAdded");
    printf("%d tests run \n", testCounter);
}