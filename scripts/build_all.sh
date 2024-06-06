#!/bin/sh

# determine folder where current script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# load corresponding environment or source modules
source ${SCRIPT_DIR}/load_env.sh

# default configuration
export CMAKE_CUDA_ARCHITECTURES=90
export CMAKE_BUILD_TYPE=Release
export USE_OMP_TARGET=0
export LIBOMPTARGET_INSTALL_PATH=/work/jk869269/repos/hpc-research/openmp/llvm-project/openmp/INSTALL

LIST_COMPUTE=(0 1)
LIST_ASYNC=(0 1)
LIST_PINNED_MEM=(0 1)
LIST_UNIFIED_MEM=(0 1)

# create directory for binaries
BINARY_DIR=${SCRIPT_DIR}/../bin
mkdir -p ${BINARY_DIR}

for comp in "${LIST_COMPUTE[@]}"
do
    for async in "${LIST_ASYNC[@]}"
    do
        for pinned in "${LIST_PINNED_MEM[@]}"
        do
            for unified in "${LIST_UNIFIED_MEM[@]}"
            do
                TMP_NAME="compute-${comp}_async-${async}_pinned-${pinned}_unified-${unified}"
                echo "===== Building ${TMP_NAME}"

                TMP_BUILD_DIR="${SCRIPT_DIR}/BUILD/${TMP_NAME}"
                TMP_BIN_DIR="${BINARY_DIR}/${TMP_NAME}"
                mkdir -p ${TMP_BUILD_DIR}
                mkdir -p ${TMP_BIN_DIR}
                cd ${TMP_BUILD_DIR}
                
                cmake   -DENABLE_COMPUTE=${comp} \
                        -DENABLE_ASYNC=${async} \
                        -DENABLE_PINNED_MEM=${pinned} \
                        -DENABLE_UNIFIED_MEM=${unified} \
                        -DHWLOC_LOCAL_INSTALL_DIR=${EBROOTHWLOC} \
                        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
                        -DUSE_OMP_TARGET=${USE_OMP_TARGET} \
                        -DLIBOMPTARGET_INSTALL_PATH=${LIBOMPTARGET_INSTALL_PATH} \
                        -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                        ${SCRIPT_DIR}/../src
                make -j8 || (echo "Error" && exit)

                # copy executable to right folder
                cp -v ${TMP_BUILD_DIR}/app/distanceBenchmark_* ${TMP_BIN_DIR}/
            done
        done
    done
done
