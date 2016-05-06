#!/bin/bash
nvcc -dryrun -arch=sm_52 -O3 -o 9-3 tiled_matmul-9-3.cu --keep 2>dryrun.out
