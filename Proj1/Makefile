G=g++ -g -Wall -lpthread -std=c++11
BIN=build/proj1
M=build/main.o
ARCH=EECS_690_Mertz_Project_1.tar.gz

build: dir main.cpp Barrier.hpp
	$(G) main.cpp -o $(BIN)

# Test program with input file
test:
	$(BIN) input.txt

mem:
	valgrind --leak-check=full --show-leak-kinds=all $(BIN) input.txt

tar:
	tar -cvf $(ARCH) \
	Makefile main.cpp Barrier.hpp \
	input.txt README.txt

dir:
	rm -rf build
	mkdir -p build

clean: dir
	rm *.tar.gz
