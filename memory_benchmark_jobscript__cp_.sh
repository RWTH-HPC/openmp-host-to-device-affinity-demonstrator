#!/usr/local_rwth/bin/zsh

#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:pascal:2
#SBATCH --exclusive
#SBATCH --job-name=bcp_-memory_benchmark
#SBATCH --output=output.%J.txt
#SBATCH --time=1-00:00:00

export MODULEPATH=$MODULEPATH:/home/co693196/.modules/modulefiles/PERSONAL
source ./env
./run_benchmark.py --config config/memory_benchmark.json --output memory_benchmark/_cp_ --binary benchmark/build/app/cp_ --no_numa_balancing
