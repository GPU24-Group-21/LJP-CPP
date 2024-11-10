
build:
	@echo "Building..."
	@g++ -o ljp main.cpp -std=c++11

clean:
	@echo "Cleaning..."
	@rm -f ljp

run:
	@chmod +x run.sh
	@./run.sh

build-run: build run

.PHONY: build clean run