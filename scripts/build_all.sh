#!/bin/sh

# load corresponding environment or source modules
source ./load_env.sh

# default configuration
export HWLOC_LOCAL_INSTALL_DIR=/home/jk869269/install/hwloc/2.5.0_gcc
export CMAKE_CUDA_ARCHITECTURES=60
export CMAKE_BUILD_TYPE=Release
export USE_OMP_TARGET=0
export LIBOMPTARGET_INSTALL_PATH=/work/jk869269/repos/hpc-hiwi/llvm-project/openmp/INSTALL

LIST_COMPUTE=(0 1)
LIST_ASYNC=(0 1)
LIST_PINNED_MEM=(0 1)
LIST_UNIFIED_MEM=(0 1)

# create directory for binaries
CUR_DIR=$PWD
BINARY_DIR=${CUR_DIR}/bin
mkdir -p ${BINARY_DIR}

for comp in "${LIST_COMPUTE[@]}"
do
    for async in "${LIST_ASYNC[@]}"
    do
        for pinned in "${LIST_PINNED_MEM[@]}"
        do
            for unified in "${LIST_UNIFIED_MEM[@]}"
            do
                TMP_NAME="c${comp}_a${async}_p${pinned}_u${unified}"
                echo "===== Building ${TMP_NAME}"

                TMP_BUILD_DIR="${CUR_DIR}/BUILD/${TMP_NAME}"
                TMP_BIN_DIR="${BINARY_DIR}/${TMP_NAME}"
                mkdir -p ${TMP_BUILD_DIR}
                mkdir -p ${TMP_BIN_DIR}
                cd ${TMP_BUILD_DIR}
                
                cmake   -DENABLE_COMPUTE=${comp} \
                        -DENABLE_ASYNC=${async} \
                        -DENABLE_PINNED_MEM=${pinned} \
                        -DENABLE_UNIFIED_MEM=${unified} \
                        -DHWLOC_LOCAL_INSTALL_DIR=${HWLOC_LOCAL_INSTALL_DIR} \
                        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
                        -DUSE_OMP_TARGET=${USE_OMP_TARGET} \
                        -DLIBOMPTARGET_INSTALL_PATH=${LIBOMPTARGET_INSTALL_PATH} \
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                        ../..
                make -j8 || (echo "Error" && exit)

                # copy executable to right folder
                cp -v ${TMP_BUILD_DIR}/app/distanceBenchmark_* ${TMP_BIN_DIR}/
            done
        done
    done
done
