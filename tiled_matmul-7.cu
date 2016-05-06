#include <stdio.h>
#include <cuda_runtime.h>
//#include <cutil.h>

#define TILE_WIDTH 64
#define WIDTH_PER_THREAD 8
#define SW TILE_WIDTH/WIDTH_PER_THREAD
#define N 2048

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

	int y8x = y*8+x;
	int x8y = x*8+y;
	int y4x = (y%4)*8+x;

	int idA_global = by + y8x;
	int idB_global = bx + x8y;

	int row = by + y*8;
	int col = bx + x*8;


	float a0, a1, a2, a3, a4, a5, a6, a7;
	float b0, b1, b2, b3, b4, b5, b6, b7;

	float c00 = 0.0; float c01 = 0.0; float c02 = 0.0; float c03 = 0.0, c04 = 0.0; float c05 = 0.0; float c06 = 0.0; float c07 = 0.0;
	float c10 = 0.0; float c11 = 0.0; float c12 = 0.0; float c13 = 0.0, c14 = 0.0; float c15 = 0.0; float c16 = 0.0; float c17 = 0.0;
	float c20 = 0.0; float c21 = 0.0; float c22 = 0.0; float c23 = 0.0, c24 = 0.0; float c25 = 0.0; float c26 = 0.0; float c27 = 0.0;
	float c30 = 0.0; float c31 = 0.0; float c32 = 0.0; float c33 = 0.0, c34 = 0.0; float c35 = 0.0; float c36 = 0.0; float c37 = 0.0;
	float c40 = 0.0; float c41 = 0.0; float c42 = 0.0; float c43 = 0.0, c44 = 0.0; float c45 = 0.0; float c46 = 0.0; float c47 = 0.0;
	float c50 = 0.0; float c51 = 0.0; float c52 = 0.0; float c53 = 0.0, c54 = 0.0; float c55 = 0.0; float c56 = 0.0; float c57 = 0.0;
	float c60 = 0.0; float c61 = 0.0; float c62 = 0.0; float c63 = 0.0, c64 = 0.0; float c65 = 0.0; float c66 = 0.0; float c67 = 0.0;
	float c70 = 0.0; float c71 = 0.0; float c72 = 0.0; float c73 = 0.0, c74 = 0.0; float c75 = 0.0; float c76 = 0.0; float c77 = 0.0;

	sA_bf[0][0*64+y8x] = tex2D(tex_A, 0, idA_global);
	sA_bf[0][1*64+y8x] = tex2D(tex_A, 1, idA_global);
	sA_bf[0][2*64+y8x] = tex2D(tex_A, 2, idA_global);
	sA_bf[0][3*64+y8x] = tex2D(tex_A, 3, idA_global);
	sA_bf[0][4*64+y8x] = tex2D(tex_A, 4, idA_global);
	sA_bf[0][5*64+y8x] = tex2D(tex_A, 5, idA_global);
	sA_bf[0][6*64+y8x] = tex2D(tex_A, 6, idA_global);
	sA_bf[0][7*64+y8x] = tex2D(tex_A, 7, idA_global);

	sB_bf[0][0*64+y8x] = tex2D(tex_B, idB_global, 0);
	sB_bf[0][1*64+y8x] = tex2D(tex_B, idB_global, 1);
	sB_bf[0][2*64+y8x] = tex2D(tex_B, idB_global, 2);
	sB_bf[0][3*64+y8x] = tex2D(tex_B, idB_global, 3);
	sB_bf[0][4*64+y8x] = tex2D(tex_B, idB_global, 4);
	sB_bf[0][5*64+y8x] = tex2D(tex_B, idB_global, 5);
	sB_bf[0][6*64+y8x] = tex2D(tex_B, idB_global, 6);
	sB_bf[0][7*64+y8x] = tex2D(tex_B, idB_global, 7);

	A_pref = sA_bf[1];
	B_pref = sB_bf[1];
	A_now  = sA_bf[0];
	B_now  = sB_bf[0];

	int track_bf = 0;

	for (int t = 8; t < k; t += 8) {

		__syncthreads();

		A_pref[0*64+y8x] = tex2D(tex_A, t  , idA_global);
		A_pref[1*64+y8x] = tex2D(tex_A, t+1, idA_global);
		A_pref[2*64+y8x] = tex2D(tex_A, t+2, idA_global);
		A_pref[3*64+y8x] = tex2D(tex_A, t+3, idA_global);
		A_pref[4*64+y8x] = tex2D(tex_A, t+4, idA_global);
		A_pref[5*64+y8x] = tex2D(tex_A, t+5, idA_global);
		A_pref[6*64+y8x] = tex2D(tex_A, t+6, idA_global);
		A_pref[7*64+y8x] = tex2D(tex_A, t+7, idA_global);

		B_pref[0*64+y8x] = tex2D(tex_B, idB_global, t  );
		B_pref[1*64+y8x] = tex2D(tex_B, idB_global, t+1);
		B_pref[2*64+y8x] = tex2D(tex_B, idB_global, t+2);
		B_pref[3*64+y8x] = tex2D(tex_B, idB_global, t+3);
		B_pref[4*64+y8x] = tex2D(tex_B, idB_global, t+4);
		B_pref[5*64+y8x] = tex2D(tex_B, idB_global, t+5);
		B_pref[6*64+y8x] = tex2D(tex_B, idB_global, t+6);
		B_pref[7*64+y8x] = tex2D(tex_B, idB_global, t+7);



		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			float a = A_now[i*64+y8x];
			float bl = B_now[i*64+y4x];
			float br = B_now[i*64+y4x+32];

			a0 = __shfl(a, 0, 8);
			a1 = __shfl(a, 1, 8);
			a2 = __shfl(a, 2, 8);
			a3 = __shfl(a, 3, 8);
			a4 = __shfl(a, 4, 8);
			a5 = __shfl(a, 5, 8);
			a6 = __shfl(a, 6, 8);
			a7 = __shfl(a, 7, 8);

			b0 = __shfl(bl, x   );
			b1 = __shfl(bl, x+8 );
			b2 = __shfl(bl, x+16);
			b3 = __shfl(bl, x+24);
			b4 = __shfl(br, x   );
			b5 = __shfl(br, x+8 );
			b6 = __shfl(br, x+16);
			b7 = __shfl(br, x+24);
			
			c00 += a0 * b0;
			c01 += a0 * b1;
			c02 += a0 * b2;
			c03 += a0 * b3;
			c04 += a0 * b4;
			c05 += a0 * b5;
			c06 += a0 * b6;
			c07 += a0 * b7;

			c10 += a1 * b0;
			c11 += a1 * b1;
			c12 += a1 * b2;
			c13 += a1 * b3;
			c14 += a1 * b4;
			c15 += a1 * b5;
			c16 += a1 * b6;
			c17 += a1 * b7;

			c20 += a2 * b0;
			c21 += a2 * b1;
			c22 += a2 * b2;
			c23 += a2 * b3;
			c24 += a2 * b4;
			c25 += a2 * b5;
			c26 += a2 * b6;
			c27 += a2 * b7;

			c30 += a3 * b0;
			c31 += a3 * b1;
			c32 += a3 * b2;
			c33 += a3 * b3;
			c34 += a3 * b4;
			c35 += a3 * b5;
			c36 += a3 * b6;
			c37 += a3 * b7;
			
			c40 += a4 * b0;
			c41 += a4 * b1;
			c42 += a4 * b2;
			c43 += a4 * b3;
			c44 += a4 * b4;
			c45 += a4 * b5;
			c46 += a4 * b6;
			c47 += a4 * b7;

			c50 += a5 * b0;
			c51 += a5 * b1;
			c52 += a5 * b2;
			c53 += a5 * b3;
			c54 += a5 * b4;
			c55 += a5 * b5;
			c56 += a5 * b6;
			c57 += a5 * b7;

			c60 += a6 * b0;
			c61 += a6 * b1;
			c62 += a6 * b2;
			c63 += a6 * b3;
			c64 += a6 * b4;
			c65 += a6 * b5;
			c66 += a6 * b6;
			c67 += a6 * b7;

			c70 += a7 * b0;
			c71 += a7 * b1;
			c72 += a7 * b2;
			c73 += a7 * b3;
			c74 += a7 * b4;
			c75 += a7 * b5;
			c76 += a7 * b6;
			c77 += a7 * b7;
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
		float a = A_now[i*64+y8x];
		float bl = B_now[i*64+y4x];
		float br = B_now[i*64+y4x+32];

		a0 = __shfl(a, 0, 8);
		a1 = __shfl(a, 1, 8);
		a2 = __shfl(a, 2, 8);
		a3 = __shfl(a, 3, 8);
		a4 = __shfl(a, 4, 8);
		a5 = __shfl(a, 5, 8);
		a6 = __shfl(a, 6, 8);
		a7 = __shfl(a, 7, 8);

		b0 = __shfl(bl, x   );
		b1 = __shfl(bl, x+8 );
		b2 = __shfl(bl, x+16);
		b3 = __shfl(bl, x+24);
		b4 = __shfl(br, x   );
		b5 = __shfl(br, x+8 );
		b6 = __shfl(br, x+16);
		b7 = __shfl(br, x+24);
		
		c00 += a0 * b0;
		c01 += a0 * b1;
		c02 += a0 * b2;
		c03 += a0 * b3;
		c04 += a0 * b4;
		c05 += a0 * b5;
		c06 += a0 * b6;
		c07 += a0 * b7;

		c10 += a1 * b0;
		c11 += a1 * b1;
		c12 += a1 * b2;
		c13 += a1 * b3;
		c14 += a1 * b4;
		c15 += a1 * b5;
		c16 += a1 * b6;
		c17 += a1 * b7;

		c20 += a2 * b0;
		c21 += a2 * b1;
		c22 += a2 * b2;
		c23 += a2 * b3;
		c24 += a2 * b4;
		c25 += a2 * b5;
		c26 += a2 * b6;
		c27 += a2 * b7;

		c30 += a3 * b0;
		c31 += a3 * b1;
		c32 += a3 * b2;
		c33 += a3 * b3;
		c34 += a3 * b4;
		c35 += a3 * b5;
		c36 += a3 * b6;
		c37 += a3 * b7;
		
		c40 += a4 * b0;
		c41 += a4 * b1;
		c42 += a4 * b2;
		c43 += a4 * b3;
		c44 += a4 * b4;
		c45 += a4 * b5;
		c46 += a4 * b6;
		c47 += a4 * b7;

		c50 += a5 * b0;
		c51 += a5 * b1;
		c52 += a5 * b2;
		c53 += a5 * b3;
		c54 += a5 * b4;
		c55 += a5 * b5;
		c56 += a5 * b6;
		c57 += a5 * b7;

		c60 += a6 * b0;
		c61 += a6 * b1;
		c62 += a6 * b2;
		c63 += a6 * b3;
		c64 += a6 * b4;
		c65 += a6 * b5;
		c66 += a6 * b6;
		c67 += a6 * b7;

		c70 += a7 * b0;
		c71 += a7 * b1;
		c72 += a7 * b2;
		c73 += a7 * b3;
		c74 += a7 * b4;
		c75 += a7 * b5;
		c76 += a7 * b6;
		c77 += a7 * b7;
	}



	surf2Dwrite(c00, surf_C, (col  )*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c01, surf_C, (col+1)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c02, surf_C, (col+2)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c03, surf_C, (col+3)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c04, surf_C, (col+4)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c05, surf_C, (col+5)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c06, surf_C, (col+6)*sizeof(float), row  , cudaBoundaryModeZero);
	surf2Dwrite(c07, surf_C, (col+7)*sizeof(float), row  , cudaBoundaryModeZero);

	surf2Dwrite(c10, surf_C, (col  )*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c11, surf_C, (col+1)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c12, surf_C, (col+2)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c13, surf_C, (col+3)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c14, surf_C, (col+4)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c15, surf_C, (col+5)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c16, surf_C, (col+6)*sizeof(float), row+1  , cudaBoundaryModeZero);
	surf2Dwrite(c17, surf_C, (col+7)*sizeof(float), row+1  , cudaBoundaryModeZero);

	surf2Dwrite(c20, surf_C, (col  )*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c21, surf_C, (col+1)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c22, surf_C, (col+2)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c23, surf_C, (col+3)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c24, surf_C, (col+4)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c25, surf_C, (col+5)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c26, surf_C, (col+6)*sizeof(float), row+2  , cudaBoundaryModeZero);
	surf2Dwrite(c27, surf_C, (col+7)*sizeof(float), row+2  , cudaBoundaryModeZero);

	surf2Dwrite(c30, surf_C, (col  )*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c31, surf_C, (col+1)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c32, surf_C, (col+2)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c33, surf_C, (col+3)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c34, surf_C, (col+4)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c35, surf_C, (col+5)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c36, surf_C, (col+6)*sizeof(float), row+3  , cudaBoundaryModeZero);
	surf2Dwrite(c37, surf_C, (col+7)*sizeof(float), row+3  , cudaBoundaryModeZero);

	surf2Dwrite(c40, surf_C, (col  )*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c41, surf_C, (col+1)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c42, surf_C, (col+2)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c43, surf_C, (col+3)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c44, surf_C, (col+4)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c45, surf_C, (col+5)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c46, surf_C, (col+6)*sizeof(float), row+4  , cudaBoundaryModeZero);
	surf2Dwrite(c47, surf_C, (col+7)*sizeof(float), row+4  , cudaBoundaryModeZero);

	surf2Dwrite(c50, surf_C, (col  )*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c51, surf_C, (col+1)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c52, surf_C, (col+2)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c53, surf_C, (col+3)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c54, surf_C, (col+4)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c55, surf_C, (col+5)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c56, surf_C, (col+6)*sizeof(float), row+5  , cudaBoundaryModeZero);
	surf2Dwrite(c57, surf_C, (col+7)*sizeof(float), row+5  , cudaBoundaryModeZero);

	surf2Dwrite(c60, surf_C, (col  )*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c61, surf_C, (col+1)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c62, surf_C, (col+2)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c63, surf_C, (col+3)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c64, surf_C, (col+4)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c65, surf_C, (col+5)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c66, surf_C, (col+6)*sizeof(float), row+6  , cudaBoundaryModeZero);
	surf2Dwrite(c67, surf_C, (col+7)*sizeof(float), row+6  , cudaBoundaryModeZero);

	surf2Dwrite(c70, surf_C, (col  )*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c71, surf_C, (col+1)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c72, surf_C, (col+2)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c73, surf_C, (col+3)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c74, surf_C, (col+4)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c75, surf_C, (col+5)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c76, surf_C, (col+6)*sizeof(float), row+7  , cudaBoundaryModeZero);
	surf2Dwrite(c77, surf_C, (col+7)*sizeof(float), row+7  , cudaBoundaryModeZero);

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


	FILE *fp = fopen("gpu.out", "w");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			fprintf(fp, "%f\n", C[i*N+j]);
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
