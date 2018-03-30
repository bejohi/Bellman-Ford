#ifndef INF236_CA2_TEST_H
#define INF236_CA2_TEST_H

#include <printf.h>
#include <stdbool.h>
#include "graphTest.h"
#include "testCompleteGraph.h"
#include "testTask1.h"

static unsigned int testCounter = 0;

static void assertTrue(bool result, char *testName) {
    if (!result) {
        printf("TEST FAILED: %s \n", testName);
    } else {
        printf("TEST Succeeded: %s \n", testName);
    }
    testCounter++;
}

void runTestSuite() {
    printf("RUN Test Suite\n");
    assertTrue(test_createGraph(), "test_createGraph");
    assertTrue(test_addEdge(), "test_addEdge");
    assertTrue(test_createCompleteGraph(), "test_createCompleteGraph");
    assertTrue(test_bellmanFord(),"test_bellmanFord");
    printf("%d tests run \n", testCounter);
}

#endif //INF236_CA2_TEST_H
