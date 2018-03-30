#ifndef INF236_CA2_TEST_H
#define INF236_CA2_TEST_H

#include <printf.h>
#include <stdbool.h>
#include "graphTest.h"
#include "task1Test.h"
#include "testCompleteGraph.h"

static unsigned int testCounter = 0;

static void assertTrue(bool result, char *testName) {
    if (!result) {
        printf("TEST FAILED: %s \n", testName);
    }
    testCounter++;
}

void runTestSuite(){
    printf("RUN Test Suite\n");
    assertTrue(test_createGraph(),"test_createGraph");
    assertTrue(test_addEdge(),"test_addEdge");
    assertTrue(testBellmanFord(),"testBellmanFord");
    assertTrue(test_createCompleteGraph(), "test_createCompleteGraph");
    printf("%d tests run \n", testCounter);
}

#endif //INF236_CA2_TEST_H
