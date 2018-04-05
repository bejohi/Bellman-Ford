cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}-fopenmp -lm -Wall -Wpedantic")

add_executable(INF236_CA2 src/main.c src/testSuite.h src/bellmanFordCompleteGraphSequential.c src/bellmanFordCompleteGraphSequential.h src/completeGraph.cpp src/completeGraph.h src/testCompleteGraph.h src/testCompleteGraph.c src/testBellmanForcCompleteGraphSequential.h src/testBellmanForcCompleteGraphSequential.c src/reportTools.c src/reportTools.h src/bellmanFordCompleteGraphCpuParallel.h src/bellmanFordCompleteGraphCpuParallel.c src/bellmanFordCompleteGraphGpuParallel.h)