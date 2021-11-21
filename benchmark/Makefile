export OMP_NUM_THREADS:=1

all: clean release run

clean:
	rm -rf *.out 2> /dev/null

release:
	nvcc -std=c++11 -O3 -arch=sm_60 -c matrix.cu -o matrix.o
	g++ -std=c++11 -Ofast -march=native -fopenmp -L /usr/local_rwth/sw/cuda/11.4.0/lib64 -lcuda -lcudart matrix_example_api.cpp matrix.o -o matrix.out

run:
	./matrix.out 1000 1
