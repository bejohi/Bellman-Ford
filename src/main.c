#include <stdbool.h>
#include "testSuite.h"

#define TEST_MODE true

int main() {
    if (TEST_MODE) {
        runTestSuite();
    }
}