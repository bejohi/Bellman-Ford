#include <stdbool.h>
#include "testSuite.h"
#include "reportTools.h"

#define TEST_MODE true
#define REPORT_MODE true

static inline Report createReportStruct(){
    Report report = {};
    report.verticesCasesSize = 7;
    report.verticesCases[0] = 10;
    report.verticesCases[1] = 100;
    report.verticesCases[2] = 1000;
    report.verticesCases[3] = 2000;
    report.verticesCases[4] = 3000;
    report.verticesCases[5] = 4000;
    report.verticesCases[6] = 4500;
    report.threadCasesSize = 7;
    report.threadCases[0] = 1;
    report.threadCases[1] = 2;
    report.threadCases[3] = 4;
    report.threadCases[4] = 8;
    report.threadCases[5] = 16;
    report.threadCases[6] = 32;
    report.threadCases[7] = 64;
    return report;

}

int main() {
    if (TEST_MODE) {
        if(!runTestSuite()){
            return -2;
        }
    }

    if(REPORT_MODE){
        unsigned int verticesSize[7] = {10,100,1000,2000,3000,4000,4500};
        printReportBellmanFordCompleteGraphSequential(verticesSize,7);
        Report report = createReportStruct();
        createReportParallelCpu(&report);

    }
}