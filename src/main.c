#include <stdbool.h>
#include "test.h"

#define TEST_MODE true

int main() {
    if (TEST_MODE) {
        runTestSuite();
    }
}