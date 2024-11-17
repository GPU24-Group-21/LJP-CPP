build:
	@nvcc -o ljp main.cu -std=c++11
	@g++ -o validator validator.cpp -std=c++11

clean:
	@echo "Cleaning..."
	@rm -f ljp*

run:
	@chmod +x run.sh
	@./run.sh

validate:
	@chmod +x validate.sh
	@./validate.sh

diff-in:
	@diff -w m.in mols.in

all: build run

.PHONY: build clean run diff-in all validate