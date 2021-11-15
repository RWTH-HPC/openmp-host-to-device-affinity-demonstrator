export OMP_NUM_THREADS:=1

all: clean release run

clean:
	rm -rf *.out 2> /dev/null

release:
	clang++ -Ofast -march=native -fopenmp matrix_example_api.cpp -o matrix.out

run:
	./matrix.out 100 1000
