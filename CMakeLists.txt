cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp -lm")

add_executable(INF236_CA2 src/graph.h src/main.c src/test.h src/graphTest.h src/task1.c src/task1Test.h src/task1.h src/task1Test.c src/graphTest.c src/graph.c)

# DEBUG in CLion: e.g. (Edge[100000000])*(Edge*)graph.adjList, on MAC: (long long(*)[30])graph.edgeList