build:
	@nvcc -o ljp main.cu -std=c++17
	@g++ -o validator validator.cpp -std=c++17

clean:
	@echo "Cleaning..."
	@rm -f ljp* validator
	@rm -rf output

run:
	@chmod +x run.sh
	@./run.sh

run-output:
	@chmod +x run.sh
	@./run.sh -v

run-cpu:
	@chmod +x run.sh
	@./run.sh -c

run-cuda:
	@chmod +x run.sh
	@./run.sh -g

run-cuda-output:
	@chmod +x run.sh
	@./run.sh -g -v

run-cpu-output:
	@chmod +x run.sh
	@./run.sh -c -v

validate:
	@chmod +x validate.sh
	@./validate.sh

plot:
	python3 plot.py
	@echo "Plot generated"

all: build run

.PHONY: build clean run all validate plot run-cpu run-cuda run-output run-cuda-output run-cpu-output