cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp -lm")

add_executable(INF236_CA2 src/task1.c src/graph.h src/testGraph.c src/testGraph.h src/main.c)

# DEBUG in CLion: e.g. (Edge[100000000])*(Edge*)graph.adjList