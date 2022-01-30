# hiwi-jan-kraus

## Compiling

Load environment using
    source env

open `benchmark/CMakeLists.txt` and change `-DCOMPUTE` and `-DASYNC` where `=1`
means activate and `=0` means deactivate.

Compile benchmark with

```shell
    mkdir benchmark/build
    cd benchmark/build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \ 
        -DCMAKE_CUDA_ARCHITECTURES=60 \ 
        -DHWLOC_LOCAL_INSTALL_DIR=/path/to/hwloc/2.5.0 \
        -DCMAKE_BUILD_TYPE=Release ..
```

## Running

Run with
    ./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]

i.e.
    ./distanceBenchmark_best 8192 10

where `best` always offloads to the closest GPU and `worst` always offloads
to the furthest GPU.

To see the GPU trace run with
    nvprof --print-gpu.trace ./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]

## Semi-Automatic Benchmarking

Execute `run_benchmark.py` using
    ./run_benchmark.py --config config/memory_benchmark.json --output ./

to generate a json output file based on the config files.
