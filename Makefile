# TODO: CLEAN-UP, add Wildcards, add Variables, add remove tool, change output path.

all: build
	./bin/bellmanFord

gpu: gbu-build
	./bin/bellmanFordCuda

gbu-build: prep
	nvcc -o bin/bellmanFordCuda src/completeGraph.c src/bellmanFordCompleteGraphGpuParallel.c -lm -O3 -Wall -Wpedantic


build: prep
	gcc -o bin/bellmanFord src/completeGraph.c src/bellmanFordCompleteGraphCpuParallel.c src/bellmanFordCompleteGraphSequential.c src/testBellmanForcCompleteGraphSequential.c src/testCompleteGraph.c src/reportTools.c src/main.c -std=c99 -fopenmp -lm -O3 -Wall -Wpedantic


prep: remove
	mkdir bin/

remove:
	rm -rf bin/