#include "graphTest.h"

bool test_createGraph() {
    // Arrange
    unsigned long numberOfVerices = 10;
    unsigned long numberOfEdges = 5;

    // Act
    Graph graph = createGraph(numberOfVerices, numberOfEdges);

    // Assert
    if (graph.edgeListSize != numberOfEdges * 3 ||
        graph.edgeList == NULL ||
        graph.numberOfEdges != numberOfEdges ||
        graph.numberOfVertices != numberOfVerices ||
        graph.edgeList[0] != EDGE_NOT_INIT ||
        graph.edgeList[graph.edgeListSize - 1] != EDGE_NOT_INIT) {
        return false;

    }

    destroyGraph(&graph);
    return true;
}

bool test_addEdge(){
    // Arrange
    Graph graph = createGraph(10, 10);

    // Act
    addEdge(&graph,0,1,10);
    addEdge(&graph,1,2,5);

    // Assert
    if(graph.edgePointer != 6 ||
       graph.edgeList[0] != 0 ||
       graph.edgeList[1] != 1 ||
       graph.edgeList[2] != 10 ||
       graph.edgeList[3] != 1 ||
       graph.edgeList[4] != 2 ||
       graph.edgeList[5] != 5){
        return false;
    }

    destroyGraph(&graph);
    return true;
}