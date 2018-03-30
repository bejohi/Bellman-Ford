#ifndef INF236_CA2_TEST_H
#define INF236_CA2_TEST_H

#include <stdio.h>
#include <stdbool.h>
#include "testCompleteGraph.h"
#include "testBellmanForcCompleteGraphSequential.h"

static unsigned int testCounter = 0;
static bool testFailed = false;

static void assertTrue(bool result, char *testName) {
    if (!result) {
        printf("TEST FAILED: %s \n", testName);
        testFailed = true;
    } else {
        printf("TEST Succeeded: %s \n", testName);
    }
    testCounter++;
}

bool runTestSuite() {
    printf("-------------\n");
    printf("RUN Test Suite\n");
    assertTrue(test_createCompleteGraph(), "test_createCompleteGraph");
    assertTrue(test_bellmanFord(),"test_bellmanFord");
    printf("%d tests run \n", testCounter);
    printf("-------------\n");
    return !testFailed;
}

#endif //INF236_CA2_TEST_H
