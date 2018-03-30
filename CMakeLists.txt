cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp -lm -O3 -Wall -Wpedantic")

add_executable(INF236_CA2 src/main.c src/testSuite.h src/bellmanFordCompleteGraphSequential.c src/bellmanFordCompleteGraphSequential.h src/completeGraph.c src/completeGraph.h src/testCompleteGraph.h src/testCompleteGraph.c src/testBellmanForcCompleteGraphSequential.h src/testBellmanForcCompleteGraphSequential.c src/reportTools.c src/reportTools.h src/bellmanFordCompleteGraphCpuParallel.h src/bellmanFordCompleteGraphCpuParallel.c)

# DEBUG in CLion: e.g. (Edge[100000000])*(Edge*)graph.adjList, on MAC: (long long(*)[30])graph.edgeList