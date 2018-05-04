G = g++ -g -std=c++11 -Wall
M = build/main.o
BIN = build/main
NAME := $(shell uname -s)
N = EECS_690_Mertz

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
	$(BIN) 256 256 256 voxeldata/aneurism.raw 1 aneurism

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

tar:
	mkdir -p $(N)
	cp $(shell find . -type f -name "*.cl") \
		$(shell find . -type f -name "*.*pp") \
		Makefile \
		$(N)
	tar -cvf $(N).tar.gz $(N)



clean: dir libdir
	rm -rf ImageWriter/*.o ImageWriter/*.so
	rm -rf $(N)
	rm -f $(N).tar.gz
