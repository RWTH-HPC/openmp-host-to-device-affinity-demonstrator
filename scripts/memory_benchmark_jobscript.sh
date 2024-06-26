#!/usr/local_rwth/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --account=supp0001
#SBATCH --partition=c23g
#SBATCH --exclusive
#SBATCH --job-name=memory_benchmark
#SBATCH --output=output.%J.txt
#SBATCH --time=08:00:00

# activate custom modules (specific to user)
module use ~/.modules

# exeucte
hostname
zsh ./run_all.sh
