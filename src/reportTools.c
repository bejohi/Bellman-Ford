#include "reportTools.h"

#define setRandomSeed() (srand((unsigned)time(NULL)))
#define randomFloat() ((float)rand()/RAND_MAX)

CompleteGraph createRandomCompleteGraph(unsigned int size){
    CompleteGraph graph = createCompleteGraph(size);
    if(graph.error){
        return graph;
    }

    setRandomSeed();
    for(unsigned int y = 0; y < size; y++){
        for(unsigned int x = 0; x < size; x++){
            graph.adjMatrix[y][x] = randomFloat();
        }
    }

    return graph;
}
