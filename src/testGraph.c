#include <stdio.h>
#include <stdbool.h>
#include "graph.h"
#include "testGraph.h"

static unsigned int testCounter = 0;


bool testInitGraph_Size0_() {
    // Arrange
    Graph graph = {};
    ULL size = 100000000;

    // Act
    initGraph(&graph, size);

    // Assert
    if (graph.size != size) {
        return false;
    }

    if (graph.adjList[0].next != NULL) {
        return false;
    }

    if (graph.adjList[size - 1].next != NULL) {
        return false;
    }

    destroyGraph(&graph);
    return true;

}

void printTestResult(bool result, char *testName) {
    if (!result) {
        printf("TEST FAILED: %s \n", testName);
    }
    testCounter++;
}

void runTestSuite() {
    printf("RUN Test Suite\n");
    printTestResult(testInitGraph_Size0_(), "testInitGraph_Size0");
    printf("%d tests run \n", testCounter);
}

