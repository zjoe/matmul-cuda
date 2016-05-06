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

	int x8 = x*8;
	int y8 = y*8;
	int y8x = y8+x;
	int x8y = x8+y;

	int idA_global = by + y8x;
	int idB_global = bx + x8y;

	int row = by + y*8;
	int col = bx + x*8;

	float a00, a01, a02, a03, a04, a05, a06, a07;
	float a10, a11, a12, a13, a14, a15, a16, a17;
	float b00, b01, b02, b03, b04, b05, b06, b07;
	float b10, b11, b12, b13, b14, b15, b16, b17;

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

		int baseA = y8;
		int baseB = x8;

		a00 = A_now[baseA    ];
		a01 = A_now[baseA + 1];
		a02 = A_now[baseA + 2];
		a03 = A_now[baseA + 3];
		a04 = A_now[baseA + 4];
		a05 = A_now[baseA + 5];
		a06 = A_now[baseA + 6];
		a07 = A_now[baseA + 7];

		b00 = B_now[baseB    ];
		b01 = B_now[baseB + 1];
		b02 = B_now[baseB + 2];
		b03 = B_now[baseB + 3];
		b04 = B_now[baseB + 4];
		b05 = B_now[baseB + 5];
		b06 = B_now[baseB + 6];
		b07 = B_now[baseB + 7];


		#pragma unroll
		for (int i = 1; i < 8; ++i) {
			int baseA = i * 64 + y8;
			int baseB = i * 64 + x8;

			if (i & 1) {
				a10 = A_now[baseA    ];
				c00 += a00 * b00;
				c01 += a00 * b01;
				c02 += a00 * b02;
				c03 += a00 * b03;
				b10 = B_now[baseB    ];
				c04 += a00 * b04;
				c05 += a00 * b05;
				c06 += a00 * b06;
				c07 += a00 * b07;

				a11 = A_now[baseA + 1];
				c10 += a01 * b00;
				c11 += a01 * b01;
				c12 += a01 * b02;
				c13 += a01 * b03;
				b11 = B_now[baseB + 1];
				c14 += a01 * b04;
				c15 += a01 * b05;
				c16 += a01 * b06;
				c17 += a01 * b07;

				a12 = A_now[baseA + 2];
				c20 += a02 * b00;
				c21 += a02 * b01;
				c22 += a02 * b02;
				c23 += a02 * b03;
				b12 = B_now[baseB + 2];
				c24 += a02 * b04;
				c25 += a02 * b05;
				c26 += a02 * b06;
				c27 += a02 * b07;

				a13 = A_now[baseA + 3];
				c30 += a03 * b00;
				c31 += a03 * b01;
				c32 += a03 * b02;
				c33 += a03 * b03;
				b13 = B_now[baseB + 3];
				c34 += a03 * b04;
				c35 += a03 * b05;
				c36 += a03 * b06;
				c37 += a03 * b07;
				
				a14 = A_now[baseA + 4];
				c40 += a04 * b00;
				c41 += a04 * b01;
				c42 += a04 * b02;
				c43 += a04 * b03;
				b14 = B_now[baseB + 4];
				c44 += a04 * b04;
				c45 += a04 * b05;
				c46 += a04 * b06;
				c47 += a04 * b07;

				a15 = A_now[baseA + 5];
				c50 += a05 * b00;
				c51 += a05 * b01;
				c52 += a05 * b02;
				c53 += a05 * b03;
				b15 = B_now[baseB + 5];
				c54 += a05 * b04;
				c55 += a05 * b05;
				c56 += a05 * b06;
				c57 += a05 * b07;

				a16 = A_now[baseA + 6];
				c60 += a06 * b00;
				c61 += a06 * b01;
				c62 += a06 * b02;
				c63 += a06 * b03;
				b16 = B_now[baseB + 6];
				c64 += a06 * b04;
				c65 += a06 * b05;
				c66 += a06 * b06;
				c67 += a06 * b07;

				a17 = A_now[baseA + 7];
				c70 += a07 * b00;
				c71 += a07 * b01;
				c72 += a07 * b02;
				c73 += a07 * b03;
				b17 = B_now[baseB + 7];
				c74 += a07 * b04;
				c75 += a07 * b05;
				c76 += a07 * b06;
				c77 += a07 * b07;
			} else {


				a00 = A_now[baseA    ];
				c00 += a10 * b10;
				c01 += a10 * b11;
				c02 += a10 * b12;
				c03 += a10 * b13;
				b00 = B_now[baseB    ];
				c04 += a10 * b14;
				c05 += a10 * b15;
				c06 += a10 * b16;
				c07 += a10 * b17;

				a01 = A_now[baseA + 1];
				c10 += a11 * b10;
				c11 += a11 * b11;
				c12 += a11 * b12;
				c13 += a11 * b13;
				b01 = B_now[baseB + 1];
				c14 += a11 * b14;
				c15 += a11 * b15;
				c16 += a11 * b16;
				c17 += a11 * b17;

				a02 = A_now[baseA + 2];
				c20 += a12 * b10;
				c21 += a12 * b11;
				c22 += a12 * b12;
				c23 += a12 * b13;
				b02 = B_now[baseB + 2];
				c24 += a12 * b14;
				c25 += a12 * b15;
				c26 += a12 * b16;
				c27 += a12 * b17;

				a03 = A_now[baseA + 3];
				c30 += a13 * b10;
				c31 += a13 * b11;
				c32 += a13 * b12;
				c33 += a13 * b13;
				b03 = B_now[baseB + 3];
				c34 += a13 * b14;
				c35 += a13 * b15;
				c36 += a13 * b16;
				c37 += a13 * b17;
				
				a04 = A_now[baseA + 4];
				c40 += a14 * b10;
				c41 += a14 * b11;
				c42 += a14 * b12;
				c43 += a14 * b13;
				b04 = B_now[baseB + 4];
				c44 += a14 * b14;
				c45 += a14 * b15;
				c46 += a14 * b16;
				c47 += a14 * b17;

				a05 = A_now[baseA + 5];
				c50 += a15 * b10;
				c51 += a15 * b11;
				c52 += a15 * b12;
				c53 += a15 * b13;
				b05 = B_now[baseB + 5];
				c54 += a15 * b14;
				c55 += a15 * b15;
				c56 += a15 * b16;
				c57 += a15 * b17;

				a06 = A_now[baseA + 6];
				c60 += a16 * b10;
				c61 += a16 * b11;
				c62 += a16 * b12;
				c63 += a16 * b13;
				b06 = B_now[baseB + 6];
				c64 += a16 * b14;
				c65 += a16 * b15;
				c66 += a16 * b16;
				c67 += a16 * b17;

				a07 = A_now[baseA + 7];
				c70 += a17 * b10;
				c71 += a17 * b11;
				c72 += a17 * b12;
				c73 += a17 * b13;
				b07 = B_now[baseB + 7];
				c74 += a17 * b14;
				c75 += a17 * b15;
				c76 += a17 * b16;
				c77 += a17 * b17;

			}
			

		}

		c00 += a10 * b10;
		c01 += a10 * b11;
		c02 += a10 * b12;
		c03 += a10 * b13;
		c04 += a10 * b14;
		c05 += a10 * b15;
		c06 += a10 * b16;
		c07 += a10 * b17;

		c10 += a11 * b10;
		c11 += a11 * b11;
		c12 += a11 * b12;
		c13 += a11 * b13;
		c14 += a11 * b14;
		c15 += a11 * b15;
		c16 += a11 * b16;
		c17 += a11 * b17;

		c20 += a12 * b10;
		c21 += a12 * b11;
		c22 += a12 * b12;
		c23 += a12 * b13;
		c24 += a12 * b14;
		c25 += a12 * b15;
		c26 += a12 * b16;
		c27 += a12 * b17;

		c30 += a13 * b10;
		c31 += a13 * b11;
		c32 += a13 * b12;
		c33 += a13 * b13;
		c34 += a13 * b14;
		c35 += a13 * b15;
		c36 += a13 * b16;
		c37 += a13 * b17;
		
		c40 += a14 * b10;
		c41 += a14 * b11;
		c42 += a14 * b12;
		c43 += a14 * b13;
		c44 += a14 * b14;
		c45 += a14 * b15;
		c46 += a14 * b16;
		c47 += a14 * b17;

		c50 += a15 * b10;
		c51 += a15 * b11;
		c52 += a15 * b12;
		c53 += a15 * b13;
		c54 += a15 * b14;
		c55 += a15 * b15;
		c56 += a15 * b16;
		c57 += a15 * b17;

		c60 += a16 * b10;
		c61 += a16 * b11;
		c62 += a16 * b12;
		c63 += a16 * b13;
		c64 += a16 * b14;
		c65 += a16 * b15;
		c66 += a16 * b16;
		c67 += a16 * b17;

		c70 += a17 * b10;
		c71 += a17 * b11;
		c72 += a17 * b12;
		c73 += a17 * b13;
		c74 += a17 * b14;
		c75 += a17 * b15;
		c76 += a17 * b16;
		c77 += a17 * b17;

		A_pref = sA_bf[track_bf];
		B_pref = sB_bf[track_bf];
		A_now  = sA_bf[1-track_bf];
		B_now  = sB_bf[1-track_bf];
		track_bf = 1 - track_bf;

	}
	__syncthreads();
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		int baseA = i * 64 + y8;
		int baseB = i * 64 + x8;

		a00 = A_now[baseA    ];
		a01 = A_now[baseA + 1];
		a02 = A_now[baseA + 2];
		a03 = A_now[baseA + 3];
		a04 = A_now[baseA + 4];
		a05 = A_now[baseA + 5];
		a06 = A_now[baseA + 6];
		a07 = A_now[baseA + 7];

		b00 = B_now[baseB    ];
		b01 = B_now[baseB + 1];
		b02 = B_now[baseB + 2];
		b03 = B_now[baseB + 3];
		b04 = B_now[baseB + 4];
		b05 = B_now[baseB + 5];
		b06 = B_now[baseB + 6];
		b07 = B_now[baseB + 7];
				
		c00 += a00 * b00;
		c01 += a00 * b01;
		c02 += a00 * b02;
		c03 += a00 * b03;
		c04 += a00 * b04;
		c05 += a00 * b05;
		c06 += a00 * b06;
		c07 += a00 * b07;

		c10 += a01 * b00;
		c11 += a01 * b01;
		c12 += a01 * b02;
		c13 += a01 * b03;
		c14 += a01 * b04;
		c15 += a01 * b05;
		c16 += a01 * b06;
		c17 += a01 * b07;

		c20 += a02 * b00;
		c21 += a02 * b01;
		c22 += a02 * b02;
		c23 += a02 * b03;
		c24 += a02 * b04;
		c25 += a02 * b05;
		c26 += a02 * b06;
		c27 += a02 * b07;

		c30 += a03 * b00;
		c31 += a03 * b01;
		c32 += a03 * b02;
		c33 += a03 * b03;
		c34 += a03 * b04;
		c35 += a03 * b05;
		c36 += a03 * b06;
		c37 += a03 * b07;
		
		c40 += a04 * b00;
		c41 += a04 * b01;
		c42 += a04 * b02;
		c43 += a04 * b03;
		c44 += a04 * b04;
		c45 += a04 * b05;
		c46 += a04 * b06;
		c47 += a04 * b07;

		c50 += a05 * b00;
		c51 += a05 * b01;
		c52 += a05 * b02;
		c53 += a05 * b03;
		c54 += a05 * b04;
		c55 += a05 * b05;
		c56 += a05 * b06;
		c57 += a05 * b07;

		c60 += a06 * b00;
		c61 += a06 * b01;
		c62 += a06 * b02;
		c63 += a06 * b03;
		c64 += a06 * b04;
		c65 += a06 * b05;
		c66 += a06 * b06;
		c67 += a06 * b07;

		c70 += a07 * b00;
		c71 += a07 * b01;
		c72 += a07 * b02;
		c73 += a07 * b03;
		c74 += a07 * b04;
		c75 += a07 * b05;
		c76 += a07 * b06;
		c77 += a07 * b07;
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
