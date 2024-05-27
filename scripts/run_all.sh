#!/bin/sh

# determine folder where current script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# load corresponding environment or source modules
source ${SCRIPT_DIR}/load_env.sh

# unlimited core files (if anything crashes)
ulimit -c unlimited

export USE_OMP_TARGET=0
if [[ "${USE_OMP_TARGET}" = "1" ]]; then
    export LIBOMPTARGET_INSTALL_PATH=/work/jk869269/repos/hpc-hiwi/llvm-project/openmp/INSTALL
    export LD_LIBRARY_PATH="$LIBOMPTARGET_INSTALL_PATH/lib:$LD_LIBRARY_PATH"
fi

LIST_COMPUTE=(0 1)
LIST_ASYNC=(0 1)
LIST_PINNED_MEM=(0 1)
LIST_UNIFIED_MEM=(0 1)
LIST_NUMA_BALANCING=(0)

# create directory for binaries
BINARY_DIR=${SCRIPT_DIR}/../bin
RESULT_DIR=${SCRIPT_DIR}/../results
mkdir -p ${RESULT_DIR}

for comp in "${LIST_COMPUTE[@]}"
do
    for async in "${LIST_ASYNC[@]}"
    do
        for pinned in "${LIST_PINNED_MEM[@]}"
        do
            for unified in "${LIST_UNIFIED_MEM[@]}"
            do
                for nb in "${LIST_NUMA_BALANCING[@]}"
                do
                    TMP_NAME="c${comp}_a${async}_p${pinned}_u${unified}"
                    TMP_NAME="compute-${comp}_async-${async}_pinned-${pinned}_unified-${unified}"
                    TMP_NAME_W_NB="nb-${nb}_compute-${comp}_async-${async}_pinned-${pinned}_unified-${unified}"

                    TMP_BIN_DIR="${BINARY_DIR}/${TMP_NAME}"
                    TMP_RESULT_DIR="${RESULT_DIR}/${TMP_NAME_W_NB}"

                    if [[ "${unified}" == "1" ]] && [[ "${pinned}" = "0" ]]; then
                        # this case does not need to be executed
                    else
                        mkdir -p ${TMP_RESULT_DIR}

                        if [[ "${nb}" = "0" ]]; then
                            echo "===== Running experiments for ${TMP_NAME_W_NB} w/o NUMA balancing"
                            python3 ${SCRIPT_DIR}/run_benchmark.py \
                                --config ${SCRIPT_DIR}/../config/memory_benchmark.json \
                                --binary ${TMP_BIN_DIR} \
                                --no_numa_balancing \
                                --output ${TMP_RESULT_DIR}
                        else
                            echo "===== Running experiments for ${TMP_NAME_W_NB} w/ NUMA balancing"
                            python3 ${SCRIPT_DIR}/run_benchmark.py \
                                --config ${CUR_DIR}/../config/memory_benchmark.json \
                                --binary ${TMP_BIN_DIR} \
                                --output ${TMP_RESULT_DIR}
                        fi
                    fi
                done
            done
        done
    done
done
