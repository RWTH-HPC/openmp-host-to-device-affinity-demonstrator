CC:=clang

LIB:=-lnuma -L/usr/local_rwth/sw/cuda/11.4.0/lib64/ -lcudart -L${HOME}/.local/install/hwloc/2.5.0/lib -lhwloc -fopenmp

all: clean debug run

debug:
	clang -g -c gpu.c -o gpu.o
	clang -g ${LIB} gpu.o main.c -o test.out
run:
	./test.out

clean:
	rm -f *.out *.o 2> /dev/null
