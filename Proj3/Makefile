G = g++ -g -std=c++11 -Wall
M = build/main.o
BIN = build/main
NAME := $(shell uname -s)

# If mac
ifeq ($(NAME), Darwin)
F = -framework OpenCL
LIB = lib/libCOGLImageWriter.dylib
EXTRA = $(G) -dynamiclib \
			-o $(LIB) \
			ImageWriter/BMPImageWriter.o \
			ImageWriter/ImageWriter.o \
			ImageWriter/JPEGImageWriter.o \
			ImageWriter/PNGImageWriter.o \
			-l png \
			-l jpeg
endif

# If linux
ifeq ($(NAME), Linux)
F = -l OpenCL
LIB = lib/libCOGLImageWriter.so
endif


build: dir main

run:
	$(BIN) 256 256 256 voxeldata/aneurism.raw 1 jpeg

gdb:
	gdb $(BIN)

# Make binary
main: main.o imglib
	$(G) $(M) $(LIB) $(F) -o $(BIN)

main.o:
	$(G) -I ImageWriter -c main.cpp -o $(M)

# Builds ImageWriter shared lib
imglib: libdir
	(cd ImageWriter; make)
	$(EXTRA)

libdir:
	rm -rf lib
	mkdir lib

dir:
	rm -rf build
	mkdir build

clean: dir libdir
	rm -rf ImageWriter/*.o ImageWriter/*.so
