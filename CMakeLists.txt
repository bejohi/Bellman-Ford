cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp -lm -O3")

add_executable(INF236_CA2 src/main.c src/testSuite.h src/task1.c src/task1.h src/completeGraph.c src/completeGraph.h src/testCompleteGraph.h src/testCompleteGraph.c src/testTask1.h src/testTask1.c)

# DEBUG in CLion: e.g. (Edge[100000000])*(Edge*)graph.adjList, on MAC: (long long(*)[30])graph.edgeList