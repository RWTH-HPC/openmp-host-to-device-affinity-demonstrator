#!/bin/sh

module purge
module load DEVELOP
module load gcc/9
module load clang/11
module load intelmpi
module load cuda/11.4
module load cmake/3.21.1
module load hwloc/2.5.0_gcc
module load python/3.9.6