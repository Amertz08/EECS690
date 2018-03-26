
build: main sample

main: main.o ImageLib
	mpic++ build/main.o ../lib/libCOGLImageReader.so -o build/main

main.o:
	mpic++ -I../Packed3DArray -I../ImageReader -c main.cpp -o build/main.o

sample: sample.o ImageLib
	g++ -o build/sample build/sample.o ../lib/libCOGLImageReader.so

sample.o: sample.c++
	g++ -c sample.c++ -std=c++11 -I../Packed3DArray -I../ImageReader -o build/sample.o

ImageLib: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

runmain:
	build/main

runsample:
	build/sample terry.jpeg

dir:
	rm -rf build
	mkdir build

clean: dir
	rm -f *.o
	rm -f ../lib/*.o ../lib/*.so
	rm -f ../ImageReader/*.o ../ImageReader/*.so
