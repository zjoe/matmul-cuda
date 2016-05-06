#include <stdio.h>
#include <cuda_runtime.h>
//#include <cutil.h>

#define TILE_WIDTH 16
#define N 2048


void err_handling(cudaError_t *err, const char *str)
{
	if (*err != cudaSuccess) {
		printf("%s\n", str);
		exit(EXIT_FAILURE);
	}
}

__global__ void matMul(const float *A, const float *B, float *C, int m, int k, int n)
{
	__shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

	//int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;

	float cVal = 0.0;
	
	for (int t = 0; t < k/TILE_WIDTH; ++t) {
		sh_A[ty][tx] = A[row*k + t*TILE_WIDTH + tx];
		sh_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k + col];
		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i) {
			cVal += sh_A[ty][i] * sh_B[i][tx];
		}
		__syncthreads();
	}
	C[row*n + col] = cVal;
}

int main(void)
{
	cudaError_t err = cudaSuccess;
	
	int m = N;
	int n = N;
	int k = N;
	
	float *A = (float*)malloc(m*k*sizeof(float));
	float *B = (float*)malloc(k*n*sizeof(float));
	float *C = (float*)malloc(m*n*sizeof(float));

	if (A == NULL || B == NULL || C == NULL) {
		printf("allocate host error!\n");
		return 1;
	}

	for (int i = 0; i < m*k; ++i) {
		A[i] = rand()/(float)RAND_MAX;
	}

	for (int i = 0; i < k*n; ++i) {
		B[i] = rand()/(float)RAND_MAX;
	}

	for (int i = 0; i < m*n; ++i) {
		C[i] = rand()/(float)RAND_MAX;
	}

	float *dev_A = NULL;
	float *dev_B = NULL;
	float *dev_C = NULL;

	err = cudaMalloc((void**)&dev_A, m*k*sizeof(float));
	err_handling(&err, "allocate devecie error A!");

	err = cudaMalloc((void**)&dev_B, k*n*sizeof(float));
	err_handling(&err, "allocate devecie error B!");

	err = cudaMalloc((void**)&dev_C, m*n*sizeof(float));
	err_handling(&err, "allocate devecie error C!");
	
	err = cudaMemcpy(dev_A, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
	err_handling(&err, "memcpy to A error!");

	err = cudaMemcpy(dev_B, B, k*n*sizeof(float), cudaMemcpyHostToDevice);
	err_handling(&err, "memcpy to B error!");

	dim3 dimGrid((m-1)/TILE_WIDTH+1, (n-1)/TILE_WIDTH+1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	matMul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, m, k, n);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float time_elapsed = 0;
	cudaEventElapsedTime(&time_elapsed, start, stop);
	printf("%fms\n", time_elapsed);

	err = cudaMemcpy(C, dev_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	err_handling(&err, "memcpy to host C error!");

	printf("%f %f\n", C[100*N+100], C[234*N+234]);

	err = cudaFree(dev_A);
	err_handling(&err, "mem free A error!");

	err = cudaFree(dev_B);
	err_handling(&err, "mem free B error!");

	err = cudaFree(dev_C);
	err_handling(&err, "mem free C error!");

	err = cudaDeviceReset();
	err_handling(&err, "device reset error!");

	return 0;
}
