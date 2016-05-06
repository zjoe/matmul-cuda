all : 9-3

% : tiled_matmul-%.cu
	nvcc -arch=sm_52 -keep --ptxas-options=-v --maxrregcount=127 -O3 -o $@ $<

#% : tiled_matmul-%.cu
#	nvcc -arch=sm_52 -ptx --ptxas-options=-v --maxrregcount=127 $<
