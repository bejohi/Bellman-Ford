# TODO: CLEAN-UP, add Wildcards, add Variables, add remove tool, change output path.
all:
	gcc completeGraph.c bellmanFordCompleteGraphCpuParallel.c bellmanFordCompleteGraphSequential.c testBellmanForcCompleteGraphSequential.c test CompleteGraph.c reportTools.c main.c -std=c99 -fopenmp -lm -O3 -Wall -Wpedantic