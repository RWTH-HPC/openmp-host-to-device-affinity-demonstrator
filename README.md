# hiwi-jan-kraus

## Prerequisits for benchmark codes

- gcc/9 required as the nvcc host compiler
- clang/11
- cuda/11.4 or similar versions
- cmake>=3.13
- hwloc/2.5.0 or similar versions
- [openmp/target-dev](https://github.com/jkravs/llvm-project/tree/target-dev/openmp) (read the readme.md there)

## Compiling

Activate or deactivate computing, async, pinned memory, unified memory in benchmark/CMakeLists.txt

Compile benchmark with

```shell
mkdir benchmark/build
cd benchmark/build
cmake -DCMAKE_CUDA_ARCHITECTURES=60 \ 
    -DHWLOC_LOCAL_INSTALL_DIR=/path/to/hwloc/2.5.0 \
    -DUSE_OMP_TARGET=1
    -DCMAKE_BUILD_TYPE=Release ..
```

## Running

Run with

```bash
./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]
```
i.e.
```bash
./distanceBenchmark_best 8192 10
```
where `best` always offloads to the closest GPU and `worst` always offloads to the furthest GPU.

To see the GPU trace run with
```bash
nvprof --print-gpu.trace ./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]
```

## Semi-Automatic Benchmarking

Execute `run_benchmark.py` using
```bash
./run_benchmark.py --config config/memory_benchmark.json --output ./
```
to generate a json output file based on the config files.
