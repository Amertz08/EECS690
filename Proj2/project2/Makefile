N=EECS_690Mertz_Project2

build: dir main sample

main: main.o ImageLib
	mpic++ build/main.o ../lib/libCOGLImageReader.so -o build/main

main.o:
	mpic++ -std=c++11 -I../Packed3DArray -I../ImageReader -c main.cpp -o build/main.o

sample: sample.o ImageLib
	g++ -o build/sample build/sample.o ../lib/libCOGLImageReader.so

sample.o: sample.c++
	g++ -c sample.c++ -std=c++11 -I../Packed3DArray -I../ImageReader -o build/sample.o

ImageLib: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

runmain:
	mpirun -np 4 \
	--allow-run-as-root \
	--mca orte_base_help_aggregate 1 \
	--mca mpi_param_check 1 \
	--mca mpi_show_handle_leaks 1 \
	build/main \
	terry.jpeg \
	hello.jpg \
	tree.jpg \
	car.jpg

runsample:
	build/sample terry.jpg

tar:
	mkdir -p $(N)
	cp main.cpp Makefile README.txt $(N)
	tar -cvf $(N).tar.gz $(N)

dir:
	rm -rf build
	mkdir build

clean: dir
	rm -rf $(N)
	rm -f $(N).tar.gz
	rm -f *.o
	rm -f ../lib/*.o ../lib/*.so
	rm -f ../ImageReader/*.o ../ImageReader/*.so
