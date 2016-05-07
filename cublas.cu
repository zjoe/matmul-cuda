#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cudaErrorHandling.h"


void matmulBatched(int m, int n, int k, int batch, const float* h_A, const float* h_B, float* h_C)
{
	
	/*********************** dev_A *******************************/
	float* d_A = NULL;
	err_handling(  cudaMalloc(&d_A, m*k*batch*sizeof(float))  );
	err_handling(  cudaMemcpy(d_A, h_A, m*k*batch*sizeof(float), cudaMemcpyHostToDevice)  );

	/*********************** dev_B *****************************/
	float* d_B = NULL;
	err_handling(  cudaMalloc(&d_B, k*n*batch*sizeof(float))  );
	err_handling(  cudaMemcpy(d_B, h_B, k*n*batch*sizeof(float), cudaMemcpyHostToDevice)  );

	/*********************** dev_C *******************************/
	float* d_C = NULL;
	err_handling(  cudaMalloc(&d_C, m*n*batch*sizeof(float))  );

	/*********************** cublas ******************************/

	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1, beta = 0;

	float **d_refA, **d_refB, **d_refC;
	err_handling(  cudaMalloc(&d_refA, batch*sizeof(float*))  );
	err_handling(  cudaMalloc(&d_refB, batch*sizeof(float*))  );
	err_handling(  cudaMalloc(&d_refC, batch*sizeof(float*))  );

	float **h_refA, **h_refB, **h_refC;
	h_refA = (float**)malloc(batch*sizeof(float*));
	h_refB = (float**)malloc(batch*sizeof(float*));
	h_refC = (float**)malloc(batch*sizeof(float*));

	for (int i = 0; i < batch; i++) {
		h_refA[i] = &d_A[i*m*k];
		h_refB[i] = &d_B[i*k*n];
		h_refC[i] = &d_C[i*m*n];
	}
	
	err_handling( cudaMemcpy(d_refA, h_refA, batch*sizeof(float*), cudaMemcpyHostToDevice) );
	err_handling( cudaMemcpy(d_refB, h_refB, batch*sizeof(float*), cudaMemcpyHostToDevice) );
	err_handling( cudaMemcpy(d_refC, h_refC, batch*sizeof(float*), cudaMemcpyHostToDevice) );

/***************************************************************************************************/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start, 0);

	cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
			   &alpha, (const float **)d_refA, m, (const float **)d_refB, n, 
			   &beta, d_refC, m, batch);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time_elapsed;
	cudaEventElapsedTime(&time_elapsed, start, stop);
	printf("time %fms\n", time_elapsed);

/****************************************************************************************************/

	err_handling( cudaMemcpy(h_C, d_C, m*n*batch*sizeof(float), cudaMemcpyDeviceToHost) );

	FILE *fout = fopen("cublas.out", "w");
	for (int b = 0; b < batch; b++) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				fprintf(fout, "%f\n", h_C[b*m*n+j*m+i]);
			}
		}
	}
	fclose(fout);

/****************************************************************************************************/

	err_handling(  cudaFree(d_A)  );
	err_handling(  cudaFree(d_B)  );
	err_handling(  cudaFree(d_C)  );
	err_handling(  cudaFree(d_refA)  );
	err_handling(  cudaFree(d_refB)  );
	err_handling(  cudaFree(d_refC)  );
	free(h_refA);
	free(h_refB);
	free(h_refC);
	cublasDestroy(handle);
}


int main(const int argc, const char *argv[])
{
	if (argc != 5) {
		printf("usage: xx.out m n k batch\n");
		return -1;
	}
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	int batch = atoi(argv[4]);
	
	int ASize = m*k*batch;
	int BSize = k*n*batch;
	int CSize = m*n*batch;

	float *A = (float*)malloc(ASize*sizeof(float));
	float *B = (float*)malloc(BSize*sizeof(float));
	float *C = (float*)malloc(CSize*sizeof(float));

	if (A == NULL || B == NULL || C == NULL)
		printf("allocate host err!\n");


	for (int i = 0; i < ASize; i++) {
		A[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < BSize; i++) {
		B[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}


/***************************************************/
	matmulBatched(m, n, k, batch, A, B, C);
/***************************************************/

	err_handling(  cudaDeviceReset()  );

	return 0;
}

