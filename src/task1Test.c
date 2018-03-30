#include "task1Test.h"

bool testBellmannFord(){
    // Arrange
    Graph graph = createGraph(10,10);
    addEdge(&graph,0,1,10);
    addEdge(&graph,1,2,5);
    addEdge(&graph,0,3,5);
    addEdge(&graph,3,2,10);
    addEdge(&graph,2,4,5);

    // Act


    destroyGraph(&graph);
    return true;
}