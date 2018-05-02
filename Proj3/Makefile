G = g++ -g -std=c++11 -Wall
M = build/main.o
BIN = build/main
NAME := $(shell uname -s)
LIB = ./lib/libCOGLImageWriter.so

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
	$(BIN) 256 256 256 ./voxeldata/aneurism.raw 1 jpeg

main: main.o imglib
	$(G) $(M) $(LIB) -o $(BIN) $(F)

main.o:
	$(G) -I ./ImageWriter -c main.cpp -o $(M)

imglib:
	(cd ./ImageWriter; make)

dir:
	rm -rf build
	mkdir build

clean: dir
