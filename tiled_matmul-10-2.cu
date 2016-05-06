#include <stdio.h>
#include <cuda_runtime.h>
//#include <cutil.h>


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
	__shared__ float sA_bf[2][8*64];
	__shared__ float sB_bf[2][8*64];
	float *A_pref, *A_now;
	float *B_pref, *B_now;

	int x = threadIdx.x;
	int y = threadIdx.y;

	int bx = blockIdx.x*64;
	int by = blockIdx.y*64;
	
	int id = y*8+x;
	int inv_id = (id%32)/4*8 + id%4 + (id < 32 ? 0 : 4);
	int glbA_id = by + inv_id;
	int glbB_id = bx + inv_id;

	int row = by + y*8;
	int col = bx + x*8;


	float a[8];
	float b[8];

	float c[8][8];

	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; j++)
			c[i][j] = 0.0;

/************************************************************************/
	
	for (int i = 0; i < 8; ++i) {
		sA_bf[0][i*64+id] = tex2D(tex_A, i, glbA_id);
		sB_bf[0][i*64+id] = tex2D(tex_B, glbB_id, i);
	}

	A_pref = sA_bf[1];
	B_pref = sB_bf[1];
	A_now  = sA_bf[0];
	B_now  = sB_bf[0];

	int track_bf = 0;

	for (int t = 8; t < k; t += 8) {

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			A_pref[i*64+id] = tex2D(tex_A, t+i, glbA_id);
			B_pref[i*64+id] = tex2D(tex_B, glbB_id, t+i);
			int base = i * 16;

			((float4*)a)[0] = ((float4*)A_now)[base+y];
			((float4*)b)[0] = ((float4*)B_now)[base+x];
			((float4*)a)[1] = ((float4*)A_now)[base+y+8];
			((float4*)b)[1] = ((float4*)B_now)[base+x+8];

			for (int ii = 0; ii < 8; ++ii)
				for (int jj = 0; jj < 8; ++jj)
					c[ii][jj] += a[ii] * b[jj];
		}

		A_pref = sA_bf[track_bf];
		B_pref = sB_bf[track_bf];
		A_now  = sA_bf[1-track_bf];
		B_now  = sB_bf[1-track_bf];
		track_bf = 1 - track_bf;

	}
	__syncthreads();


	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		int base = i * 16;

		((float4*)a)[0] = ((float4*)A_now)[base+y];
		((float4*)b)[0] = ((float4*)B_now)[base+x];
		((float4*)a)[1] = ((float4*)A_now)[base+y+8];
		((float4*)b)[1] = ((float4*)B_now)[base+x+8];

		for (int ii = 0; ii < 8; ++ii)
			for (int jj = 0; jj < 8; ++jj)
				c[ii][jj] += a[ii] * b[jj];
		
	}
	
/******************************** write back ****************************************/
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			surf2Dwrite(c[i][j], surf_C, (col+j)*sizeof(float), row+i  , cudaBoundaryModeZero);
		}
	}

}
int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("usage: ./xxx m n k\n");
		return -1;
	}
	
	cudaError_t err = cudaSuccess;
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	
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



	dim3 dimGrid((n-1)/64+1, (m-1)/64+1, 1);
	dim3 dimBlock(8, 8, 1);

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
