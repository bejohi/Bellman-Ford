#include <stdio.h>
#include <stdbool.h>
#include "graph.h"
#include "testGraph.h"

static unsigned int testCounter = 0;


bool test_initGraph_size100000000() {
    // Arrange
    Graph graph = {};
    ULL size = 100000000; // 100.000.000

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

bool test_addEdgeInBothDirections_correctEdge(){
    // Arrange
    Graph graph = {};
    ULL size = 10;
    initGraph(&graph,size);

    // Act
    addEdgeInBothDirections(&graph,0,1,100.5);

    // Assert
    if(graph.adjList[0].next == NULL || graph.adjList[0].next->weight != 100.5 || graph.adjList[0].next->index != 1){
        return false;
    }

    if(graph.adjList[1].next == NULL || graph.adjList[1].next->weight != 100.5 || graph.adjList[1].next->index != 0){
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
    printTestResult(test_initGraph_size100000000(), "test_initGraph_size100000000");
    printTestResult(test_addEdgeInBothDirections_correctEdge(),"test_addEdgeInBothDirections_correctEdge");
    printf("%d tests run \n", testCounter);
}

