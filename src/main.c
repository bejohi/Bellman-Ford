#include "test.h"
#include <stdbool.h>
#define TEST_MODE true

int main(){
    if(TEST_MODE){
        runTestSuite();
    }
}