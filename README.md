# OpenMP Host-to-Device Affinity Showcase

## Prerequisits for benchmark codes

- gcc/9 or higher required as the nvcc host compiler
- clang/11 (the modified version is based on version 11)
- cuda/11.4 or similar versions
- cmake>=3.13
- hwloc/2.5.0
- For LLVM variant: [extended LLVM openmp runtime](https://github.com/RWTH-HPC/llvm-project/tree/thread-to-device-affinity/openmp) (read the `README.md` there)

**Note:** Convenience scripts for compiling and running are sourcing `load_env.sh` which is loading the required modules. Customize that script according to your cluster environment  

## Compiling
* This benchmark provides a CMake build system
* There a several flags that can be for the compilation

| Flag | Description |
|---|---|
| `ENABLE_COMPUTE` | Whether to enable computation in the offloaded kernel or hardly do anything focusing on the memory transfers and offload latency only |
| `ENABLE_ASYNC` | Whether to use asynchornous offloading |
| `ENABLE_PINNED_MEM` | Whether to use CUDA pinned memory |
| `ENABLE_UNIFIED_MEM` | Whether to use CUDA unified memory. Only used when `ENABLE_PINNED_MEM=1` |
| `HWLOC_LOCAL_INSTALL_DIR` | Install directory for hwloc |
| `CMAKE_CUDA_ARCHITECTURES` | CUDA architectures (compute-capabilities) to use |
| `USE_OMP_TARGET` | Whether to use LLVM target offloading (`1`) or CUDA prototype (`0`) |
| `LIBOMPTARGET_INSTALL_PATH` | Installation directory of customized openmp runtime with affinity-aware libomptarget extension |

* Versions can be build the following way:
```bash
# create BUILD directory and jump into
mkdir -p BUILD && cd BUILD
# create Makefiles with cmake
cmake \
    -DENABLE_COMPUTE=0 \
    -DENABLE_ASYNC=0 \
    -DENABLE_PINNED_MEM=0 \
    -DENABLE_UNIFIED_MEM=0 \
    -DHWLOC_LOCAL_INSTALL_DIR=<path/to/hwloc/install> \
    -DCMAKE_CUDA_ARCHITECTURES=60 \
    -DUSE_OMP_TARGET=1 \
    -DLIBOMPTARGET_INSTALL_PATH=<path/to/hwloc/install> \
    -DCMAKE_BUILD_TYPE=Release \
    ../src
# build benchmarks
make
```
* This will build two executables
  * `distanceBenchmark_best`: select always the closest GPU for thread
  * `distanceBenchmark_worst`: select always the GPU furthest away from thread
* Additionally, there is a script (`scripts/build_all.sh`) to build all variants that are currently supported. Just set the desired paths and common variables in the script

## Running
* If you want to run the LLVM variant, make sure that additional paths are set to use the customized runtime
```bash
# prepand library paths
export LD_LIBRARY_PATH="$LIBOMPTARGET_INSTALL_PATH/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$LIBOMPTARGET_INSTALL_PATH/lib:$LIBRARY_PATH"
# prepend include paths
export INCLUDE="$LIBOMPTARGET_INSTALL_PATH/include:$INCLUDE"
export CPATH="$LIBOMPTARGET_INSTALL_PATH/include:$CPATH"
export C_INCLUDE_PATH="$LIBOMPTARGET_INSTALL_PATH/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$LIBOMPTARGET_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH"
```
* Syntax of program execution is:
```bash
# Syntax:
./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]
# Example:
./distanceBenchmark_best 8192 10
```
* To see the GPU trace run with
```bash
nvprof --print-gpu.trace ./distanceBenchmark_(best|worst) [matrix_size] [number_of_tasks]
```

## Semi-Automatic Benchmarking
* This repo contains additional script to automatically run a series of benchmark executions based on a configuration file
* These scripts are called `scripts/run_all.sh` (main entry point) and `scripts/run_benchmark.py` (automate single benchmark run based on config)

## Evaluation
* To gather data from the result files of the benchmark run execute
```bash
# print help and list parameters
python3 scripts/evaluate.py -h
# run evaluation in data in result directory
python3 scripts/evaluate.py -s <result_dir>
```
* This script will create plots for all variants executed
* For each variant there will be two plots
  * one focusing on the small problem sizes
  * one focusing on the larger problem sizes
* Specify `--plot_threshold <thr>` where to split plots between small and large. Default is 1024
  