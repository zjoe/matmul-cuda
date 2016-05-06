all : 0 1 2 3 4 5 6 7 8 9 10 10-1 9-1 9-2 10-2 9-3 9-4

% : tiled_matmul-%.cu
	nvcc -arch=sm_52 --ptxas-options=-v --maxrregcount=127 -O3 -o $@ $<

#% : tiled_matmul-%.cu
#	nvcc -arch=sm_52 -ptx --ptxas-options=-v --maxrregcount=127 $<
