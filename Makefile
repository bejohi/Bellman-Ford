# TODO: CLEAN-UP, add Wildcards, add Variables, add remove tool, change output path.
prep:
	mkdir bin/

all:
	gcc -o bin/bellmanFord bin/src/completeGraph.c src/bellmanFordCompleteGraphCpuParallel.c src/bellmanFordCompleteGraphSequential.c src/testBellmanForcCompleteGraphSequential.c src/testCompleteGraph.c src/reportTools.c src/main.c -std=c99 -fopenmp -lm -O3 -Wall -Wpedantic

remove:
	rm -rf bin/