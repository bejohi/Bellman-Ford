//
// Created by Jonas Hielscher on 30.03.18.
//

#ifndef INF236_CA2_TEST_H
#define INF236_CA2_TEST_H

#include <printf.h>
#include <stdbool.h>
#include "graphTest.h"

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
    printf("%d tests run \n", testCounter);
}

#endif //INF236_CA2_TEST_H
