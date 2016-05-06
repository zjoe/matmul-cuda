#include <stdio.h>
#include <cuda_runtime.h>
//#include <cutil.h>

#define TILE_WIDTH 64
#define WIDTH_PER_THREAD 4
#define SW TILE_WIDTH/WIDTH_PER_THREAD
#define N 961

texture<float, 2, cudaReadModeElementType> tex_A;
texture<float, 2, cudaReadModeElementType> tex_B;
surface<void, 2> surf_C;

void err_handling(cudaError_t *err, const char *str)
{
	if (*err != cudaSuccess) {
		printf("%s\n", str);
		exit(EXIT_FAILURE);
	}
}

__global__ void matMul(const float *A, const float *B, float *C, int m, int k, int n)
{
	__shared__ float sh_A00[SW][SW], sh_A01[SW][SW], sh_A02[SW][SW], sh_A03[SW][SW];
	__shared__ float sh_A10[SW][SW], sh_A11[SW][SW], sh_A12[SW][SW], sh_A13[SW][SW];
	__shared__ float sh_A20[SW][SW], sh_A21[SW][SW], sh_A22[SW][SW], sh_A23[SW][SW];
	__shared__ float sh_A30[SW][SW], sh_A31[SW][SW], sh_A32[SW][SW], sh_A33[SW][SW];

	__shared__ float sh_B00[SW][SW], sh_B01[SW][SW], sh_B02[SW][SW], sh_B03[SW][SW];
	__shared__ float sh_B10[SW][SW], sh_B11[SW][SW], sh_B12[SW][SW], sh_B13[SW][SW];
	__shared__ float sh_B20[SW][SW], sh_B21[SW][SW], sh_B22[SW][SW], sh_B23[SW][SW];
	__shared__ float sh_B30[SW][SW], sh_B31[SW][SW], sh_B32[SW][SW], sh_B33[SW][SW];

	int x = threadIdx.x;
	int y = threadIdx.y;
	int tx = x*WIDTH_PER_THREAD;
	int ty = y*WIDTH_PER_THREAD;

	int row = blockIdx.y*TILE_WIDTH + ty;
	int col = blockIdx.x*TILE_WIDTH + tx;


	float a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
	float b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33;
	float c00 = 0.0; float c01 = 0.0; float c02 = 0.0; float c03 = 0.0;
	float c10 = 0.0; float c11 = 0.0; float c12 = 0.0; float c13 = 0.0;
	float c20 = 0.0; float c21 = 0.0; float c22 = 0.0; float c23 = 0.0;
	float c30 = 0.0; float c31 = 0.0; float c32 = 0.0; float c33 = 0.0;

	for (int t = 0; t < k; t += TILE_WIDTH) {
		sh_A00[y][x] = tex2D(tex_A, t+tx  , row);
		sh_A01[y][x] = tex2D(tex_A, t+tx+1, row);
		sh_A02[y][x] = tex2D(tex_A, t+tx+2, row);
		sh_A03[y][x] = tex2D(tex_A, t+tx+3, row);
		sh_A10[y][x] = tex2D(tex_A, t+tx  , row+1);
		sh_A11[y][x] = tex2D(tex_A, t+tx+1, row+1);
		sh_A12[y][x] = tex2D(tex_A, t+tx+2, row+1);
		sh_A13[y][x] = tex2D(tex_A, t+tx+3, row+1);
		sh_A20[y][x] = tex2D(tex_A, t+tx  , row+2);
		sh_A21[y][x] = tex2D(tex_A, t+tx+1, row+2);
		sh_A22[y][x] = tex2D(tex_A, t+tx+2, row+2);
		sh_A23[y][x] = tex2D(tex_A, t+tx+3, row+2);
		sh_A30[y][x] = tex2D(tex_A, t+tx  , row+3);
		sh_A31[y][x] = tex2D(tex_A, t+tx+1, row+3);
		sh_A32[y][x] = tex2D(tex_A, t+tx+2, row+3);
		sh_A33[y][x] = tex2D(tex_A, t+tx+3, row+3);

		sh_B00[y][x] = tex2D(tex_B, col  , t+ty);
		sh_B01[y][x] = tex2D(tex_B, col+1, t+ty);
		sh_B02[y][x] = tex2D(tex_B, col+2, t+ty);
		sh_B03[y][x] = tex2D(tex_B, col+3, t+ty);
		sh_B10[y][x] = tex2D(tex_B, col  , t+ty+1);
		sh_B11[y][x] = tex2D(tex_B, col+1, t+ty+1);
		sh_B12[y][x] = tex2D(tex_B, col+2, t+ty+1);
		sh_B13[y][x] = tex2D(tex_B, col+3, t+ty+1);
		sh_B20[y][x] = tex2D(tex_B, col  , t+ty+2);
		sh_B21[y][x] = tex2D(tex_B, col+1, t+ty+2);
		sh_B22[y][x] = tex2D(tex_B, col+2, t+ty+2);
		sh_B23[y][x] = tex2D(tex_B, col+3, t+ty+2);
		sh_B30[y][x] = tex2D(tex_B, col  , t+ty+3);
		sh_B31[y][x] = tex2D(tex_B, col+1, t+ty+3);
		sh_B32[y][x] = tex2D(tex_B, col+2, t+ty+3);
		sh_B33[y][x] = tex2D(tex_B, col+3, t+ty+3);

		__syncthreads();

		int ii = x;
		for (int i = 0; i < TILE_WIDTH; i += WIDTH_PER_THREAD) {
			ii %= 16;
			a00 = sh_A00[y][ii]; a01 = sh_A01[y][ii]; a02 = sh_A02[y][ii]; a03 = sh_A03[y][ii];
			a10 = sh_A10[y][ii]; a11 = sh_A11[y][ii]; a12 = sh_A12[y][ii]; a13 = sh_A13[y][ii];
			a20 = sh_A20[y][ii]; a21 = sh_A21[y][ii]; a22 = sh_A22[y][ii]; a23 = sh_A23[y][ii];
			a30 = sh_A30[y][ii]; a31 = sh_A31[y][ii]; a32 = sh_A32[y][ii]; a33 = sh_A33[y][ii];

			b00 = sh_B00[ii][x]; b01 = sh_B01[ii][x]; b02 = sh_B02[ii][x]; b03 = sh_B03[ii][x]; 
			b10 = sh_B10[ii][x]; b11 = sh_B11[ii][x]; b12 = sh_B12[ii][x]; b13 = sh_B13[ii][x];
			b20 = sh_B20[ii][x]; b21 = sh_B21[ii][x]; b22 = sh_B22[ii][x]; b23 = sh_B23[ii][x];
			b30 = sh_B30[ii][x]; b31 = sh_B31[ii][x]; b32 = sh_B32[ii][x]; b33 = sh_B33[ii][x];

			c00 += a00*b00 + a01*b10 + a02*b20 + a03*b30;
			c01 += a00*b01 + a01*b11 + a02*b21 + a03*b31;
			c02 += a00*b02 + a01*b12 + a02*b22 + a03*b32;
			c03 += a00*b03 + a01*b13 + a02*b23 + a03*b33;

			c10 += a10*b00 + a11*b10 + a12*b20 + a13*b30;
			c11 += a10*b01 + a11*b11 + a12*b21 + a13*b31;
			c12 += a10*b02 + a11*b12 + a12*b22 + a13*b32;
			c13 += a10*b03 + a11*b13 + a12*b23 + a13*b33;

			c20 += a20*b00 + a21*b10 + a22*b20 + a23*b30;
			c21 += a20*b01 + a21*b11 + a22*b21 + a23*b31;
			c22 += a20*b02 + a21*b12 + a22*b22 + a23*b32;
			c23 += a20*b03 + a21*b13 + a22*b23 + a23*b33;

			c30 += a30*b00 + a31*b10 + a32*b20 + a33*b30;
			c31 += a30*b01 + a31*b11 + a32*b21 + a33*b31;
			c32 += a30*b02 + a31*b12 + a32*b22 + a33*b32;
			c33 += a30*b03 + a31*b13 + a32*b23 + a33*b33;

			++ii;
		}
		__syncthreads();
	}

	surf2Dwrite(c00, surf_C, (col  )*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c01, surf_C, (col+1)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c02, surf_C, (col+2)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c03, surf_C, (col+3)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c10, surf_C, (col  )*sizeof(float), row+1, cudaBoundaryModeZero);
	surf2Dwrite(c11, surf_C, (col+1)*sizeof(float), row+1, cudaBoundaryModeZero);
	surf2Dwrite(c12, surf_C, (col+2)*sizeof(float), row+1, cudaBoundaryModeZero);
	surf2Dwrite(c13, surf_C, (col+3)*sizeof(float), row+1, cudaBoundaryModeZero);
	surf2Dwrite(c20, surf_C, (col  )*sizeof(float), row+2, cudaBoundaryModeZero);
	surf2Dwrite(c21, surf_C, (col+1)*sizeof(float), row+2, cudaBoundaryModeZero);
	surf2Dwrite(c22, surf_C, (col+2)*sizeof(float), row+2, cudaBoundaryModeZero);
	surf2Dwrite(c23, surf_C, (col+3)*sizeof(float), row+2, cudaBoundaryModeZero);
	surf2Dwrite(c30, surf_C, (col  )*sizeof(float), row+3, cudaBoundaryModeZero);
	surf2Dwrite(c31, surf_C, (col+1)*sizeof(float), row+3, cudaBoundaryModeZero);
	surf2Dwrite(c32, surf_C, (col+2)*sizeof(float), row+3, cudaBoundaryModeZero);
	surf2Dwrite(c33, surf_C, (col+3)*sizeof(float), row+3, cudaBoundaryModeZero);

}

int main(void)
{
	cudaError_t err = cudaSuccess;
	
	int m = N;
	int n = N;
	int k = N;
	m = 961;
	n = 128;
	k = 128;
	
	float *A = (float*)malloc(m*k*sizeof(float));
	float *B = (float*)malloc(k*n*sizeof(float));
	float *C = (float*)malloc(m*n*sizeof(float));

	if (A == NULL || B == NULL || C == NULL) {
		printf("allocate host error!\n");
		return 1;
	}

	for (int i = 0; i < m*k; ++i) {
		A[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}

	for (int i = 0; i < k*n; ++i) {
		B[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
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

	cudaChannelFormatDesc ADesc = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc BDesc = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc CDesc = cudaCreateChannelDesc<float>();
	cudaArray *A_array, *B_array, *C_array;
	cudaMallocArray(&A_array, &ADesc, k, m);
	cudaMallocArray(&B_array, &BDesc, n, k);
	cudaMallocArray(&C_array, &CDesc, n, m, cudaArraySurfaceLoadStore);
	cudaMemcpyToArray(A_array, 0, 0, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(B_array, 0, 0, B, k*n*sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTextureToArray(tex_A, A_array);
	cudaBindTextureToArray(tex_B, B_array);
	cudaBindSurfaceToArray(surf_C, C_array);
	
	tex_A.addressMode[0] = cudaAddressModeBorder;
	tex_A.addressMode[1] = cudaAddressModeBorder;

	tex_B.addressMode[0] = cudaAddressModeBorder;
	tex_B.addressMode[1] = cudaAddressModeBorder;



	dim3 dimGrid((n-1)/TILE_WIDTH+1, (m-1)/TILE_WIDTH+1, 1);
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

	err = cudaMemcpyFromArray(C, C_array, 0, 0, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	//err = cudaMemcpy(C, dev_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	err_handling(&err, "memcpy to host C error!");

	printf("%f %f\n", C[100*n+100], C[234*n+234]);

	FILE *fp = fopen("gpu.out", "w");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			fprintf(fp, "%f\n", C[i*n+j]);
		}
	}
	fclose(fp);

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
