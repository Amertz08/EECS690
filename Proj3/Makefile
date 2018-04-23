G = g++ -g -std=c++11 -Wall
M = build/main.o
BIN = build/main
NAME := $(shell uname -s)

# If mac
ifeq ($(NAME), Darwin)
F = -framework OpenCL
endif

# If linux
ifeq ($(NAME), Linux)
F = -l OpenCL
endif


build: dir main

run:
	$(BIN)

main: main.o
	$(G) $(M) -o $(BIN) $(F)

main.o:
	$(G) -c main.cpp -o $(M)

dir:
	rm -rf build
	mkdir build

clean: dir
