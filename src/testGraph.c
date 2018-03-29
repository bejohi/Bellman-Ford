#include <stdio.h>
#include <stdbool.h>
#include "graph.h"
#include "testGraph.h"

static unsigned int testCounter = 0;

static bool test_createGraph_sizeIs10000_graphCorrectCreated(){
    // Arrange
    unsigned int size = 10000;
    // Act
    Graph graph = createGraph(size);

    // Assert
    if(graph.size != size){
        return false;
    }

    if(graph.adjMatrix[0][0] != 0){
        return false;
    }

    if(graph.adjMatrix[size-1][size-1] != 0){
        return false;
    }
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
    printf("%d tests run \n", testCounter);
}