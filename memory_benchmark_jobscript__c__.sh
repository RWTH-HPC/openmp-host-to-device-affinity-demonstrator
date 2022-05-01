#!/usr/local_rwth/bin/zsh

#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:pascal:2
#SBATCH --exclusive
#SBATCH --job-name=bc__-memory_benchmark
#SBATCH --output=output.%J.txt
#SBATCH --time=1-02:00:00

export MODULEPATH=$MODULEPATH:$HOME/.modules/modulefiles/PERSONAL
source ./env
./run_benchmark.py --config config/memory_benchmark.json --output memory_benchmark/_c__ --binary benchmark/build/app/c__ --no_numa_balancing
