# TODO: CLEAN-UP, add Wildcards, add Variables, add remove tool, change output path.

all: build
	./bin/bellmanFord

gpu: gpu-build
	./bin/bellmanFordCuda

gpu-build: prep
	nvcc -o bin/bellmanFordCuda src/bellmanFordCompleteGraphGpuParallel.cu -std=c++11 -lm


build: prep
	gcc -o bin/bellmanFord src/completeGraph.c src/bellmanFordCompleteGraphCpuParallel.c src/bellmanFordCompleteGraphSequential.c src/testBellmanForcCompleteGraphSequential.c src/testCompleteGraph.c src/reportTools.c src/main.c -fopenmp -std=c99 -lm -O3 -Wall -Wpedantic


prep: remove
	mkdir bin/

remove:
	rm -rf bin/