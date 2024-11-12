build-cuda:
	@nvcc -o ljp main.cu -std=c++11

build:
	@echo "Building..."
	@g++ -o ljp main.cpp -std=c++11

build-cuda:
	@echo "Building with CUDA..."
	@nvcc -o ljp main.cu

clean:
	@echo "Cleaning..."
	@rm -f ljp

run:
	@chmod +x run.sh
	@./run.sh

diff-in:
	@diff -w m.in mols.in

all: build run

all-cuda: build-cuda run

.PHONY: build clean run