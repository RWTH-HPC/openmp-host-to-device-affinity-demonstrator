#!/bin/sh

# load corresponding environment or source modules
source ./env

LIST_COMPUTE=(0 1)
LIST_ASYNC=(0 1)
LIST_PINNED_MEM=(0 1)
LIST_UNIFIED_MEM=(0 1)
LIST_NUMA_BALANCING=(0 1)

# create directory for binaries
CUR_DIR=$PWD
BINARY_DIR=${CUR_DIR}/benchmark/bin
RESULT_DIR=${CUR_DIR}/results
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
                    TMP_NAME_W_NB="nb${nb}_c${comp}_a${async}_p${pinned}_u${unified}"
                    echo "===== Running experiments for ${TMP_NAME_W_NB}"

                    TMP_BIN_DIR="${BINARY_DIR}/${TMP_NAME}"
                    TMP_RESULT_DIR="${RESULT_DIR}/${TMP_NAME_W_NB}"
                    mkdir -p ${TMP_RESULT_DIR}

                    if [ "${nb}" = "0" ]; then
                        python3 ${CUR_DIR}/run_benchmark.py \
                            --config ${CUR_DIR}/config/memory_benchmark.json \
                            --binary ${TMP_BIN_DIR} \
                            --output ${TMP_RESULT_DIR} \
                            --no_numa_balancing
                    else
                        python3 ${CUR_DIR}/run_benchmark.py \
                            --config ${CUR_DIR}/config/memory_benchmark.json \
                            --binary ${TMP_BIN_DIR} \
                            --output ${TMP_RESULT_DIR}
                    fi
                done
            done
        done
    done
done
