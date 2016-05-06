#include <stdio.h>
#include <cuda_runtime.h>
//#include <cutil.h>

#define TILE_WIDTH 64
#define WIDTH_PER_THREAD 4
#define SW TILE_WIDTH/WIDTH_PER_THREAD
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
	__shared__ float sh_A00[SW][SW];
	__shared__ float sh_A01[SW][SW];
	__shared__ float sh_A02[SW][SW];
	__shared__ float sh_A03[SW][SW];
	__shared__ float sh_A10[SW][SW];
	__shared__ float sh_A11[SW][SW];
	__shared__ float sh_A12[SW][SW];
	__shared__ float sh_A13[SW][SW];
	__shared__ float sh_A20[SW][SW];
	__shared__ float sh_A21[SW][SW];
	__shared__ float sh_A22[SW][SW];
	__shared__ float sh_A23[SW][SW];
	__shared__ float sh_A30[SW][SW];
	__shared__ float sh_A31[SW][SW];
	__shared__ float sh_A32[SW][SW];
	__shared__ float sh_A33[SW][SW];
	__shared__ float sh_B00[SW][SW];
	__shared__ float sh_B01[SW][SW];
	__shared__ float sh_B02[SW][SW];
	__shared__ float sh_B03[SW][SW];
	__shared__ float sh_B10[SW][SW];
	__shared__ float sh_B11[SW][SW];
	__shared__ float sh_B12[SW][SW];
	__shared__ float sh_B13[SW][SW];
	__shared__ float sh_B20[SW][SW];
	__shared__ float sh_B21[SW][SW];
	__shared__ float sh_B22[SW][SW];
	__shared__ float sh_B23[SW][SW];
	__shared__ float sh_B30[SW][SW];
	__shared__ float sh_B31[SW][SW];
	__shared__ float sh_B32[SW][SW];
	__shared__ float sh_B33[SW][SW];

	int x = threadIdx.x;
	int y = threadIdx.y;
	int tx = x*WIDTH_PER_THREAD;
	int ty = y*WIDTH_PER_THREAD;

	int row = blockIdx.y*TILE_WIDTH + ty;
	int col = blockIdx.x*TILE_WIDTH + tx;

	float c00 = 0.0;
	float c01 = 0.0;
	float c02 = 0.0;
	float c03 = 0.0;
	float c10 = 0.0;
	float c11 = 0.0;
	float c12 = 0.0;
	float c13 = 0.0;
	float c20 = 0.0;
	float c21 = 0.0;
	float c22 = 0.0;
	float c23 = 0.0;
	float c30 = 0.0;
	float c31 = 0.0;
	float c32 = 0.0;
	float c33 = 0.0;

	float a00 = 0.0;
	float a01 = 0.0;
	float a02 = 0.0;
	float a03 = 0.0;
	float a10 = 0.0;
	float a11 = 0.0;
	float a12 = 0.0;
	float a13 = 0.0;
	float a20 = 0.0;
	float a21 = 0.0;
	float a22 = 0.0;
	float a23 = 0.0;
	float a30 = 0.0;
	float a31 = 0.0;
	float a32 = 0.0;
	float a33 = 0.0;

	float b00 = 0.0;
	float b01 = 0.0;
	float b02 = 0.0;
	float b03 = 0.0;
	float b10 = 0.0;
	float b11 = 0.0;
	float b12 = 0.0;
	float b13 = 0.0;
	float b20 = 0.0;
	float b21 = 0.0;
	float b22 = 0.0;
	float b23 = 0.0;
	float b30 = 0.0;
	float b31 = 0.0;
	float b32 = 0.0;
	float b33 = 0.0;
	
	for (int t = 0; t < k; t += TILE_WIDTH) {
		sh_A00[y][x] = A[row*k     + t + tx];
		sh_A01[y][x] = A[row*k     + t + tx+1];
		sh_A02[y][x] = A[row*k     + t + tx+2];
		sh_A03[y][x] = A[row*k     + t + tx+3];
		sh_A10[y][x] = A[(row+1)*k + t + tx];
		sh_A11[y][x] = A[(row+1)*k + t + tx+1];
		sh_A12[y][x] = A[(row+1)*k + t + tx+2];
		sh_A13[y][x] = A[(row+1)*k + t + tx+3];
		sh_A20[y][x] = A[(row+2)*k + t + tx];
		sh_A21[y][x] = A[(row+2)*k + t + tx+1];
		sh_A22[y][x] = A[(row+2)*k + t + tx+2];
		sh_A23[y][x] = A[(row+2)*k + t + tx+3];
		sh_A30[y][x] = A[(row+3)*k + t + tx];
		sh_A31[y][x] = A[(row+3)*k + t + tx+1];
		sh_A32[y][x] = A[(row+3)*k + t + tx+2];
		sh_A33[y][x] = A[(row+3)*k + t + tx+3];


		sh_B00[y][x] = B[(t+ty)*k   + col];
		sh_B01[y][x] = B[(t+ty)*k   + col+1];
		sh_B02[y][x] = B[(t+ty)*k   + col+2];
		sh_B03[y][x] = B[(t+ty)*k   + col+3];
		sh_B10[y][x] = B[(t+ty+1)*k + col];
		sh_B11[y][x] = B[(t+ty+1)*k + col+1];
		sh_B12[y][x] = B[(t+ty+1)*k + col+2];
		sh_B13[y][x] = B[(t+ty+1)*k + col+3];
		sh_B20[y][x] = B[(t+ty+2)*k + col];
		sh_B21[y][x] = B[(t+ty+2)*k + col+1];
		sh_B22[y][x] = B[(t+ty+2)*k + col+2];
		sh_B23[y][x] = B[(t+ty+2)*k + col+3];
		sh_B30[y][x] = B[(t+ty+3)*k + col];
		sh_B31[y][x] = B[(t+ty+3)*k + col+1];
		sh_B32[y][x] = B[(t+ty+3)*k + col+2];
		sh_B33[y][x] = B[(t+ty+3)*k + col+3];
		__syncthreads();

		int ii = x;
		for (int i = 0; i < TILE_WIDTH; i += WIDTH_PER_THREAD) {
			ii %= 16;
			a00 = sh_A00[y][ii];
			a01 = sh_A01[y][ii];
			a10 = sh_A10[y][ii];
			a11 = sh_A11[y][ii];

			b00 = sh_B00[ii][x];
			b01 = sh_B01[ii][x];
			b10 = sh_B10[ii][x];
			b11 = sh_B11[ii][x];

			c00 += a00*b00 + a01*b10;
			c01 += a00*b01 + a01*b11;
			c10 += a10*b00 + a11*b10;
			c11 += a10*b01 + a11*b11;

/*******************************************************************/
			a22 = sh_A22[y][ii];
			a23 = sh_A23[y][ii];
			a32 = sh_A32[y][ii];
			a33 = sh_A33[y][ii];

			b22 = sh_B22[ii][x];
			b23 = sh_B23[ii][x];
			b32 = sh_B32[ii][x];
			b33 = sh_B33[ii][x];

			c22 += a22*b22 + a23*b32;
			c23 += a22*b23 + a23*b33;
			c32 += a32*b22 + a33*b32;
			c33 += a32*b23 + a33*b33;

/*******************************************************************/
			a02 = sh_A02[y][ii];
			a03 = sh_A03[y][ii];
			a12 = sh_A12[y][ii];
			a13 = sh_A13[y][ii];

			b20 = sh_B20[ii][x];
			b21 = sh_B21[ii][x];
			b30 = sh_B30[ii][x];
			b31 = sh_B31[ii][x];

			c00 += a02*b20 + a03*b30;
			c01 += a02*b21 + a03*b31;
			c10 += a12*b20 + a13*b30;
			c11 += a12*b21 + a13*b31;

			c02 += a02*b22 + a03*b32;
			c03 += a02*b23 + a03*b33;
			c12 += a12*b22 + a13*b32;
			c13 += a12*b23 + a13*b33;

			c20 += a22*b20 + a23*b30;
			c21 += a22*b21 + a23*b31;
			c30 += a32*b20 + a33*b30;
			c31 += a32*b21 + a33*b31;
/*******************************************************************/

			a20 = sh_A20[y][ii];
			a21 = sh_A21[y][ii];
			a30 = sh_A30[y][ii];
			a31 = sh_A31[y][ii];

			b02 = sh_B02[ii][x];
			b03 = sh_B03[ii][x];
			b12 = sh_B12[ii][x];
			b13 = sh_B13[ii][x];

			c22 += a20*b02 + a21*b12;
			c23 += a20*b03 + a21*b13;
			c32 += a30*b02 + a31*b12;
			c33 += a30*b03 + a31*b13;

			c20 += a20*b00 + a21*b10;
			c21 += a20*b01 + a21*b11;
			c30 += a30*b00 + a31*b10;
			c31 += a30*b01 + a31*b11;

			c02 += a00*b02 + a01*b12;
			c03 += a00*b03 + a01*b13;
			c12 += a10*b02 + a11*b12;
			c13 += a10*b03 + a11*b13;
/*******************************************************************/

			++ii;
		}
		__syncthreads();
	}
	C[row*n     + col]   = c00;
	C[row*n     + col+1] = c01;
	C[row*n     + col+2] = c02;
	C[row*n     + col+3] = c03;
	C[(row+1)*n + col]   = c10;
	C[(row+1)*n + col+1] = c11;
	C[(row+1)*n + col+2] = c12;
	C[(row+1)*n + col+3] = c13;
	C[(row+2)*n + col]   = c20;
	C[(row+2)*n + col+1] = c21;
	C[(row+2)*n + col+2] = c22;
	C[(row+2)*n + col+3] = c23;
	C[(row+3)*n + col]   = c30;
	C[(row+3)*n + col+1] = c31;
	C[(row+3)*n + col+2] = c32;
	C[(row+3)*n + col+3] = c33;
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
	dim3 dimBlock(TILE_WIDTH/WIDTH_PER_THREAD, TILE_WIDTH/WIDTH_PER_THREAD, 1);

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
