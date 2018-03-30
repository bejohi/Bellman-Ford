#include <stdbool.h>
#include "testSuite.h"
#include "reportTools.h"

#define TEST_MODE true
#define REPORT_MODE true

int main() {
    if (TEST_MODE) {
        if(!runTestSuite()){
            return -2;
        }
    }

    if(REPORT_MODE){
        unsigned int verticesSize[4] = {10,100,1000,10000};
        printReportBellmanFordCompleteGraphSequential(verticesSize,4);

    }
}