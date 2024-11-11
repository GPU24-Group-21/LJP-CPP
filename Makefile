
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

build-run: build run

.PHONY: build clean run