#!/bin/bash

nvcc -arch=sm_52 -O3 -keep -maxrregcount=127 -o 9-3 tiled_matmul-9-3.cu
