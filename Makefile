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

all-output: build run-output

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  build: Compile the project"
	@echo "  clean: Clean the project"
	@echo "  run: Run the project"
	@echo "  run-output: Run the project and output the results"
	@echo "  run-cpu: Run the project on CPU"
	@echo "  run-cuda: Run the project on CUDA"
	@echo "  run-cuda-output: Run the project on CUDA and output the results"
	@echo "  run-cpu-output: Run the project on CPU and output the results"
	@echo "  validate: Validate the output"
	@echo "  plot: Generate the plot"
	@echo "  all: Build and run the project"
	@echo "  all-output: Build and run the project and output the results"

.PHONY: build clean run all validate plot run-cpu run-cuda run-output run-cuda-output run-cpu-output all-output help