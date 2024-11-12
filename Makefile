TARGET = ljp

all: build run

all-cuda: build-cuda run

build-cuda:
	@nvcc -o ljp main.cu -std=c++11

build:
	@echo "Building..."
	@g++ -o ljp main.cpp -std=c++11

clean:
	@echo "Cleaning..."
	@rm -f ljp

run:
	@chmod +x run.sh
	@./run.sh

diff-in:
	@diff -w m.in mols.in

.PHONY: build clean run