build:
	@nvcc -o ljp main.cu -std=c++11
	@g++ -o validator validator.cpp -std=c++11

clean:
	@echo "Cleaning..."
	@rm -f ljp* validator
	@rm -rf output

run:
	@chmod +x run.sh
	@./run.sh

run-cpu:
	@chmod +x run.sh
	@./run.sh -c

run-cuda:
	@chmod +x run.sh
	@./run.sh -g

validate:
	@chmod +x validate.sh
	@./validate.sh

plot:
	python3 plot.py
	@echo "Plot generated"

all: build run

.PHONY: build clean run all validate plot run-cpu run-cuda