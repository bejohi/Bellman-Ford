#include <stdbool.h>
#include "testSuite.h"
#include "reportTools.h"

#define TEST_MODE true
#define REPORT_MODE true

static inline Report createReportStruct(){
    Report report = {.verticesCasesSize = 7, .threadCasesSize = 8, .numberOfRuns = 10};
    report.verticesCases[0] = 10;
    report.verticesCases[1] = 100;
    report.verticesCases[2] = 1000;
    report.verticesCases[3] = 2000;
    report.verticesCases[4] = 4000;
    report.verticesCases[5] = 8000;
    report.verticesCases[6] = 10000;
    report.threadCases[0] = 1;
    report.threadCases[1] = 2;
    report.threadCases[2] = 4;
    report.threadCases[3] = 8;
    report.threadCases[4] = 16;
    report.threadCases[5] = 32;
    report.threadCases[6] = 36;
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
        Report report = createReportStruct();
        createReport(&report);
        //createReportParallelCpu(&report);
    }
}