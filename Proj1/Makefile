G=g++ -g -Wall -std=c++11
BIN=build/proj1
M=build/main.o
ARCH=EECS_690_Mertz_Project_1.tar.gz

build: main.o
	$(G) $(M) -o $(BIN)

main.o: dir
	$(G) -c main.cpp -o $(M)

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
