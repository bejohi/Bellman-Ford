#ifndef INF236_CA2_GRAPHTEST_H
#define INF236_CA2_GRAPHTEST_H

#include <stdbool.h>
#include "graph.h"

bool test_createGraph() {
    // Arrange
    unsigned long numberOfVerices = 10;
    unsigned long numberOfEdges = 10;

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

#endif //INF236_CA2_GRAPHTEST_H
