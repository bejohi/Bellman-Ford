cmake_minimum_required(VERSION 3.10)
project(INF236_CA2 C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fopenmp -lm")

add_executable(INF236_CA2 src/task1.c)