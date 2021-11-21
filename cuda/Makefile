

all: clear debug run

clear:
	rm -rf *.out *.o 2> /dev/null

debug:
	nvcc -arch=sm_60 -std=c++11 -c matrix.cu -o matrix.o
	g++ -std=c++11 -L/usr/local_rwth/sw/cuda/11.4.0/lib64/ -lcudart -lcuda main.cpp matrix.o -o matrix.out

run:
	./matrix.out
