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
	__shared__ float sA00[SW][SW], sA01[SW][SW], sA02[SW][SW], sA03[SW][SW], sA04[SW][SW], sA05[SW][SW], sA06[SW][SW], sA07[SW][SW];
	__shared__ float sA10[SW][SW], sA11[SW][SW], sA12[SW][SW], sA13[SW][SW], sA14[SW][SW], sA15[SW][SW], sA16[SW][SW], sA17[SW][SW];
	__shared__ float sA20[SW][SW], sA21[SW][SW], sA22[SW][SW], sA23[SW][SW], sA24[SW][SW], sA25[SW][SW], sA26[SW][SW], sA27[SW][SW];
	__shared__ float sA30[SW][SW], sA31[SW][SW], sA32[SW][SW], sA33[SW][SW], sA34[SW][SW], sA35[SW][SW], sA36[SW][SW], sA37[SW][SW];

	__shared__ float sA40[SW][SW], sA41[SW][SW], sA42[SW][SW], sA43[SW][SW], sA44[SW][SW], sA45[SW][SW], sA46[SW][SW], sA47[SW][SW];
	__shared__ float sA50[SW][SW], sA51[SW][SW], sA52[SW][SW], sA53[SW][SW], sA54[SW][SW], sA55[SW][SW], sA56[SW][SW], sA57[SW][SW];
	__shared__ float sA60[SW][SW], sA61[SW][SW], sA62[SW][SW], sA63[SW][SW], sA64[SW][SW], sA65[SW][SW], sA66[SW][SW], sA67[SW][SW];
	__shared__ float sA70[SW][SW], sA71[SW][SW], sA72[SW][SW], sA73[SW][SW], sA74[SW][SW], sA75[SW][SW], sA76[SW][SW], sA77[SW][SW];

	__shared__ float sB00[SW][SW], sB01[SW][SW], sB02[SW][SW], sB03[SW][SW], sB04[SW][SW], sB05[SW][SW], sB06[SW][SW], sB07[SW][SW];
	__shared__ float sB10[SW][SW], sB11[SW][SW], sB12[SW][SW], sB13[SW][SW], sB14[SW][SW], sB15[SW][SW], sB16[SW][SW], sB17[SW][SW];
	__shared__ float sB20[SW][SW], sB21[SW][SW], sB22[SW][SW], sB23[SW][SW], sB24[SW][SW], sB25[SW][SW], sB26[SW][SW], sB27[SW][SW];
	__shared__ float sB30[SW][SW], sB31[SW][SW], sB32[SW][SW], sB33[SW][SW], sB34[SW][SW], sB35[SW][SW], sB36[SW][SW], sB37[SW][SW];

	__shared__ float sB40[SW][SW], sB41[SW][SW], sB42[SW][SW], sB43[SW][SW], sB44[SW][SW], sB45[SW][SW], sB46[SW][SW], sB47[SW][SW];
	__shared__ float sB50[SW][SW], sB51[SW][SW], sB52[SW][SW], sB53[SW][SW], sB54[SW][SW], sB55[SW][SW], sB56[SW][SW], sB57[SW][SW];
	__shared__ float sB60[SW][SW], sB61[SW][SW], sB62[SW][SW], sB63[SW][SW], sB64[SW][SW], sB65[SW][SW], sB66[SW][SW], sB67[SW][SW];
	__shared__ float sB70[SW][SW], sB71[SW][SW], sB72[SW][SW], sB73[SW][SW], sB74[SW][SW], sB75[SW][SW], sB76[SW][SW], sB77[SW][SW];

	int x = threadIdx.x;
	int y = threadIdx.y;
	int tx = x*WIDTH_PER_THREAD;
	int ty = y*WIDTH_PER_THREAD;

	int row = blockIdx.y*TILE_WIDTH + ty;
	int col = blockIdx.x*TILE_WIDTH + tx;


	float a00, a01, a02, a03, a04, a05, a06, a07;
	float a10, a11, a12, a13, a14, a15, a16, a17;
	float a20, a21, a22, a23, a24, a25, a26, a27;
	float a30, a31, a32, a33, a34, a35, a36, a37;
	float a40, a41, a42, a43, a44, a45, a46, a47;
	float a50, a51, a52, a53, a54, a55, a56, a57;
	float a60, a61, a62, a63, a64, a65, a66, a67;
	float a70, a71, a72, a73, a74, a75, a76, a77;

	float b00, b01, b02, b03, b04, b05, b06, b07;
	float b10, b11, b12, b13, b14, b15, b16, b17;
	float b20, b21, b22, b23, b24, b25, b26, b27;
	float b30, b31, b32, b33, b34, b35, b36, b37;
	float b40, b41, b42, b43, b44, b45, b46, b47;
	float b50, b51, b52, b53, b54, b55, b56, b57;
	float b60, b61, b62, b63, b64, b65, b66, b67;
	float b70, b71, b72, b73, b74, b75, b76, b77;

	float c00 = 0.0; float c01 = 0.0; float c02 = 0.0; float c03 = 0.0, c04 = 0.0; float c05 = 0.0; float c06 = 0.0; float c07 = 0.0;
	float c10 = 0.0; float c11 = 0.0; float c12 = 0.0; float c13 = 0.0, c14 = 0.0; float c15 = 0.0; float c16 = 0.0; float c17 = 0.0;
	float c20 = 0.0; float c21 = 0.0; float c22 = 0.0; float c23 = 0.0, c24 = 0.0; float c25 = 0.0; float c26 = 0.0; float c27 = 0.0;
	float c30 = 0.0; float c31 = 0.0; float c32 = 0.0; float c33 = 0.0, c34 = 0.0; float c35 = 0.0; float c36 = 0.0; float c37 = 0.0;
	float c40 = 0.0; float c41 = 0.0; float c42 = 0.0; float c43 = 0.0, c44 = 0.0; float c45 = 0.0; float c46 = 0.0; float c47 = 0.0;
	float c50 = 0.0; float c51 = 0.0; float c52 = 0.0; float c53 = 0.0, c54 = 0.0; float c55 = 0.0; float c56 = 0.0; float c57 = 0.0;
	float c60 = 0.0; float c61 = 0.0; float c62 = 0.0; float c63 = 0.0, c64 = 0.0; float c65 = 0.0; float c66 = 0.0; float c67 = 0.0;
	float c70 = 0.0; float c71 = 0.0; float c72 = 0.0; float c73 = 0.0, c74 = 0.0; float c75 = 0.0; float c76 = 0.0; float c77 = 0.0;

	for (int t = 0; t < k; t += TILE_WIDTH) {
		sA00[y][x] = tex2D(tex_A, t+tx  , row);
		sA01[y][x] = tex2D(tex_A, t+tx+1, row);
		sA02[y][x] = tex2D(tex_A, t+tx+2, row);
		sA03[y][x] = tex2D(tex_A, t+tx+3, row);
		sA04[y][x] = tex2D(tex_A, t+tx+4, row);
		sA05[y][x] = tex2D(tex_A, t+tx+5, row);
		sA06[y][x] = tex2D(tex_A, t+tx+6, row);
		sA07[y][x] = tex2D(tex_A, t+tx+7, row);

		sA10[y][x] = tex2D(tex_A, t+tx  , row+1);
		sA11[y][x] = tex2D(tex_A, t+tx+1, row+1);
		sA12[y][x] = tex2D(tex_A, t+tx+2, row+1);
		sA13[y][x] = tex2D(tex_A, t+tx+3, row+1);
		sA14[y][x] = tex2D(tex_A, t+tx+4, row+1);
		sA15[y][x] = tex2D(tex_A, t+tx+5, row+1);
		sA16[y][x] = tex2D(tex_A, t+tx+6, row+1);
		sA17[y][x] = tex2D(tex_A, t+tx+7, row+1);

		sA20[y][x] = tex2D(tex_A, t+tx  , row+2);
		sA21[y][x] = tex2D(tex_A, t+tx+1, row+2);
		sA22[y][x] = tex2D(tex_A, t+tx+2, row+2);
		sA23[y][x] = tex2D(tex_A, t+tx+3, row+2);
		sA24[y][x] = tex2D(tex_A, t+tx+4, row+2);
		sA25[y][x] = tex2D(tex_A, t+tx+5, row+2);
		sA26[y][x] = tex2D(tex_A, t+tx+6, row+2);
		sA27[y][x] = tex2D(tex_A, t+tx+7, row+2);

		sA30[y][x] = tex2D(tex_A, t+tx  , row+3);
		sA31[y][x] = tex2D(tex_A, t+tx+1, row+3);
		sA32[y][x] = tex2D(tex_A, t+tx+2, row+3);
		sA33[y][x] = tex2D(tex_A, t+tx+3, row+3);
		sA34[y][x] = tex2D(tex_A, t+tx+4, row+3);
		sA35[y][x] = tex2D(tex_A, t+tx+5, row+3);
		sA36[y][x] = tex2D(tex_A, t+tx+6, row+3);
		sA37[y][x] = tex2D(tex_A, t+tx+7, row+3);
		
		sA40[y][x] = tex2D(tex_A, t+tx  , row+4);
		sA41[y][x] = tex2D(tex_A, t+tx+1, row+4);
		sA42[y][x] = tex2D(tex_A, t+tx+2, row+4);
		sA43[y][x] = tex2D(tex_A, t+tx+3, row+4);
		sA44[y][x] = tex2D(tex_A, t+tx+4, row+4);
		sA45[y][x] = tex2D(tex_A, t+tx+5, row+4);
		sA46[y][x] = tex2D(tex_A, t+tx+6, row+4);
		sA47[y][x] = tex2D(tex_A, t+tx+7, row+4);
		
		sA50[y][x] = tex2D(tex_A, t+tx  , row+5);
		sA51[y][x] = tex2D(tex_A, t+tx+1, row+5);
		sA52[y][x] = tex2D(tex_A, t+tx+2, row+5);
		sA53[y][x] = tex2D(tex_A, t+tx+3, row+5);
		sA54[y][x] = tex2D(tex_A, t+tx+4, row+5);
		sA55[y][x] = tex2D(tex_A, t+tx+5, row+5);
		sA56[y][x] = tex2D(tex_A, t+tx+6, row+5);
		sA57[y][x] = tex2D(tex_A, t+tx+7, row+5);
		
		sA60[y][x] = tex2D(tex_A, t+tx  , row+6);
		sA61[y][x] = tex2D(tex_A, t+tx+1, row+6);
		sA62[y][x] = tex2D(tex_A, t+tx+2, row+6);
		sA63[y][x] = tex2D(tex_A, t+tx+3, row+6);
		sA64[y][x] = tex2D(tex_A, t+tx+4, row+6);
		sA65[y][x] = tex2D(tex_A, t+tx+5, row+6);
		sA66[y][x] = tex2D(tex_A, t+tx+6, row+6);
		sA67[y][x] = tex2D(tex_A, t+tx+7, row+6);
		
		sA70[y][x] = tex2D(tex_A, t+tx  , row+7);
		sA71[y][x] = tex2D(tex_A, t+tx+1, row+7);
		sA72[y][x] = tex2D(tex_A, t+tx+2, row+7);
		sA73[y][x] = tex2D(tex_A, t+tx+3, row+7);
		sA74[y][x] = tex2D(tex_A, t+tx+4, row+7);
		sA75[y][x] = tex2D(tex_A, t+tx+5, row+7);
		sA76[y][x] = tex2D(tex_A, t+tx+6, row+7);
		sA77[y][x] = tex2D(tex_A, t+tx+7, row+7);

		sB00[y][x] = tex2D(tex_B, col  , t+ty);
		sB01[y][x] = tex2D(tex_B, col+1, t+ty);
		sB02[y][x] = tex2D(tex_B, col+2, t+ty);
		sB03[y][x] = tex2D(tex_B, col+3, t+ty);
		sB04[y][x] = tex2D(tex_B, col+4, t+ty);
		sB05[y][x] = tex2D(tex_B, col+5, t+ty);
		sB06[y][x] = tex2D(tex_B, col+6, t+ty);
		sB07[y][x] = tex2D(tex_B, col+7, t+ty);

		sB10[y][x] = tex2D(tex_B, col  , t+ty+1);
		sB11[y][x] = tex2D(tex_B, col+1, t+ty+1);
		sB12[y][x] = tex2D(tex_B, col+2, t+ty+1);
		sB13[y][x] = tex2D(tex_B, col+3, t+ty+1);
		sB14[y][x] = tex2D(tex_B, col+4, t+ty+1);
		sB15[y][x] = tex2D(tex_B, col+5, t+ty+1);
		sB16[y][x] = tex2D(tex_B, col+6, t+ty+1);
		sB17[y][x] = tex2D(tex_B, col+7, t+ty+1);

		sB20[y][x] = tex2D(tex_B, col  , t+ty+2);
		sB21[y][x] = tex2D(tex_B, col+1, t+ty+2);
		sB22[y][x] = tex2D(tex_B, col+2, t+ty+2);
		sB23[y][x] = tex2D(tex_B, col+3, t+ty+2);
		sB24[y][x] = tex2D(tex_B, col+4, t+ty+2);
		sB25[y][x] = tex2D(tex_B, col+5, t+ty+2);
		sB26[y][x] = tex2D(tex_B, col+6, t+ty+2);
		sB27[y][x] = tex2D(tex_B, col+7, t+ty+2);

		sB30[y][x] = tex2D(tex_B, col  , t+ty+3);
		sB31[y][x] = tex2D(tex_B, col+1, t+ty+3);
		sB32[y][x] = tex2D(tex_B, col+2, t+ty+3);
		sB33[y][x] = tex2D(tex_B, col+3, t+ty+3);
		sB34[y][x] = tex2D(tex_B, col+4, t+ty+3);
		sB35[y][x] = tex2D(tex_B, col+5, t+ty+3);
		sB36[y][x] = tex2D(tex_B, col+6, t+ty+3);
		sB37[y][x] = tex2D(tex_B, col+7, t+ty+3);

		sB40[y][x] = tex2D(tex_B, col  , t+ty+4);
		sB41[y][x] = tex2D(tex_B, col+1, t+ty+4);
		sB42[y][x] = tex2D(tex_B, col+2, t+ty+4);
		sB43[y][x] = tex2D(tex_B, col+3, t+ty+4);
		sB44[y][x] = tex2D(tex_B, col+4, t+ty+4);
		sB45[y][x] = tex2D(tex_B, col+5, t+ty+4);
		sB46[y][x] = tex2D(tex_B, col+6, t+ty+4);
		sB47[y][x] = tex2D(tex_B, col+7, t+ty+4);

		sB50[y][x] = tex2D(tex_B, col  , t+ty+5);
		sB51[y][x] = tex2D(tex_B, col+1, t+ty+5);
		sB52[y][x] = tex2D(tex_B, col+2, t+ty+5);
		sB53[y][x] = tex2D(tex_B, col+3, t+ty+5);
		sB54[y][x] = tex2D(tex_B, col+4, t+ty+5);
		sB55[y][x] = tex2D(tex_B, col+5, t+ty+5);
		sB56[y][x] = tex2D(tex_B, col+6, t+ty+5);
		sB57[y][x] = tex2D(tex_B, col+7, t+ty+5);

		sB60[y][x] = tex2D(tex_B, col  , t+ty+6);
		sB61[y][x] = tex2D(tex_B, col+1, t+ty+6);
		sB62[y][x] = tex2D(tex_B, col+2, t+ty+6);
		sB63[y][x] = tex2D(tex_B, col+3, t+ty+6);
		sB64[y][x] = tex2D(tex_B, col+4, t+ty+6);
		sB65[y][x] = tex2D(tex_B, col+5, t+ty+6);
		sB66[y][x] = tex2D(tex_B, col+6, t+ty+6);
		sB67[y][x] = tex2D(tex_B, col+7, t+ty+6);

		sB70[y][x] = tex2D(tex_B, col  , t+ty+7);
		sB71[y][x] = tex2D(tex_B, col+1, t+ty+7);
		sB72[y][x] = tex2D(tex_B, col+2, t+ty+7);
		sB73[y][x] = tex2D(tex_B, col+3, t+ty+7);
		sB74[y][x] = tex2D(tex_B, col+4, t+ty+7);
		sB75[y][x] = tex2D(tex_B, col+5, t+ty+7);
		sB76[y][x] = tex2D(tex_B, col+6, t+ty+7);
		sB77[y][x] = tex2D(tex_B, col+7, t+ty+7);

		__syncthreads();

		int ii = x;
		for (int i = 0; i < TILE_WIDTH; i += WIDTH_PER_THREAD) {
			ii %= 8;
			a00 = sA00[y][ii]; a01 = sA01[y][ii]; a02 = sA02[y][ii]; a03 = sA03[y][ii];
			a10 = sA10[y][ii]; a11 = sA11[y][ii]; a12 = sA12[y][ii]; a13 = sA13[y][ii];
			a20 = sA20[y][ii]; a21 = sA21[y][ii]; a22 = sA22[y][ii]; a23 = sA23[y][ii];
			a30 = sA30[y][ii]; a31 = sA31[y][ii]; a32 = sA32[y][ii]; a33 = sA33[y][ii];
			a04 = sA04[y][ii]; a05 = sA05[y][ii]; a06 = sA06[y][ii]; a07 = sA07[y][ii];
			a14 = sA14[y][ii]; a15 = sA15[y][ii]; a16 = sA16[y][ii]; a17 = sA17[y][ii];
			a24 = sA24[y][ii]; a25 = sA25[y][ii]; a26 = sA26[y][ii]; a27 = sA27[y][ii];
			a34 = sA34[y][ii]; a35 = sA35[y][ii]; a36 = sA36[y][ii]; a37 = sA37[y][ii];

			a40 = sA40[y][ii]; a41 = sA41[y][ii]; a42 = sA42[y][ii]; a43 = sA43[y][ii];
			a50 = sA50[y][ii]; a51 = sA51[y][ii]; a52 = sA52[y][ii]; a53 = sA53[y][ii];
			a60 = sA60[y][ii]; a61 = sA61[y][ii]; a62 = sA62[y][ii]; a63 = sA63[y][ii];
			a70 = sA70[y][ii]; a71 = sA71[y][ii]; a72 = sA72[y][ii]; a73 = sA73[y][ii];
			a44 = sA44[y][ii]; a45 = sA45[y][ii]; a46 = sA46[y][ii]; a47 = sA47[y][ii];
			a54 = sA54[y][ii]; a55 = sA55[y][ii]; a56 = sA56[y][ii]; a57 = sA57[y][ii];
			a64 = sA64[y][ii]; a65 = sA65[y][ii]; a66 = sA66[y][ii]; a67 = sA67[y][ii];
			a74 = sA74[y][ii]; a75 = sA75[y][ii]; a76 = sA76[y][ii]; a77 = sA77[y][ii];

			b00 = sB00[ii][x]; b01 = sB01[ii][x]; b02 = sB02[ii][x]; b03 = sB03[ii][x]; 
			b10 = sB10[ii][x]; b11 = sB11[ii][x]; b12 = sB12[ii][x]; b13 = sB13[ii][x];
			b20 = sB20[ii][x]; b21 = sB21[ii][x]; b22 = sB22[ii][x]; b23 = sB23[ii][x];
			b30 = sB30[ii][x]; b31 = sB31[ii][x]; b32 = sB32[ii][x]; b33 = sB33[ii][x];
			b04 = sB04[ii][x]; b05 = sB05[ii][x]; b06 = sB06[ii][x]; b07 = sB07[ii][x]; 
			b14 = sB14[ii][x]; b15 = sB15[ii][x]; b16 = sB16[ii][x]; b17 = sB17[ii][x];
			b24 = sB24[ii][x]; b25 = sB25[ii][x]; b26 = sB26[ii][x]; b27 = sB27[ii][x];
			b34 = sB34[ii][x]; b35 = sB35[ii][x]; b36 = sB36[ii][x]; b37 = sB37[ii][x];

			b40 = sB40[ii][x]; b41 = sB41[ii][x]; b42 = sB42[ii][x]; b43 = sB43[ii][x]; 
			b50 = sB50[ii][x]; b51 = sB51[ii][x]; b52 = sB52[ii][x]; b53 = sB53[ii][x];
			b60 = sB60[ii][x]; b61 = sB61[ii][x]; b62 = sB62[ii][x]; b63 = sB63[ii][x];
			b70 = sB70[ii][x]; b71 = sB71[ii][x]; b72 = sB72[ii][x]; b73 = sB73[ii][x];
			b44 = sB44[ii][x]; b45 = sB45[ii][x]; b46 = sB46[ii][x]; b47 = sB47[ii][x]; 
			b54 = sB54[ii][x]; b55 = sB55[ii][x]; b56 = sB56[ii][x]; b57 = sB57[ii][x];
			b64 = sB64[ii][x]; b65 = sB65[ii][x]; b66 = sB66[ii][x]; b67 = sB67[ii][x];
			b74 = sB74[ii][x]; b75 = sB75[ii][x]; b76 = sB76[ii][x]; b77 = sB77[ii][x];

			c00 += a00*b00 + a01*b10 + a02*b20 + a03*b30 + a04*b40 + a05*b50 + a06*b60 + a07*b70;
			c01 += a00*b01 + a01*b11 + a02*b21 + a03*b31 + a04*b41 + a05*b51 + a06*b61 + a07*b71;
			c02 += a00*b02 + a01*b12 + a02*b22 + a03*b32 + a04*b42 + a05*b52 + a06*b62 + a07*b72;
			c03 += a00*b03 + a01*b13 + a02*b23 + a03*b33 + a04*b43 + a05*b53 + a06*b63 + a07*b73;
			c04 += a00*b04 + a01*b14 + a02*b24 + a03*b34 + a04*b44 + a05*b54 + a06*b64 + a07*b74;
			c05 += a00*b05 + a01*b15 + a02*b25 + a03*b35 + a04*b45 + a05*b55 + a06*b65 + a07*b75;
			c06 += a00*b06 + a01*b16 + a02*b26 + a03*b36 + a04*b46 + a05*b56 + a06*b66 + a07*b76;
			c07 += a00*b07 + a01*b17 + a02*b27 + a03*b37 + a04*b47 + a05*b57 + a06*b67 + a07*b77;

			c10 += a10*b00 + a11*b10 + a12*b20 + a13*b30 + a14*b40 + a15*b50 + a16*b60 + a17*b70;
			c11 += a10*b01 + a11*b11 + a12*b21 + a13*b31 + a14*b41 + a15*b51 + a16*b61 + a17*b71;
			c12 += a10*b02 + a11*b12 + a12*b22 + a13*b32 + a14*b42 + a15*b52 + a16*b62 + a17*b72;
			c13 += a10*b03 + a11*b13 + a12*b23 + a13*b33 + a14*b43 + a15*b53 + a16*b63 + a17*b73;
			c14 += a10*b04 + a11*b14 + a12*b24 + a13*b34 + a14*b44 + a15*b54 + a16*b64 + a17*b74;
			c15 += a10*b05 + a11*b15 + a12*b25 + a13*b35 + a14*b45 + a15*b55 + a16*b65 + a17*b75;
			c16 += a10*b06 + a11*b16 + a12*b26 + a13*b36 + a14*b46 + a15*b56 + a16*b66 + a17*b76;
			c17 += a10*b07 + a11*b17 + a12*b27 + a13*b37 + a14*b47 + a15*b57 + a16*b67 + a17*b77;

			c20 += a20*b00 + a21*b10 + a22*b20 + a23*b30 + a24*b40 + a25*b50 + a26*b60 + a27*b70;
			c21 += a20*b01 + a21*b11 + a22*b21 + a23*b31 + a24*b41 + a25*b51 + a26*b61 + a27*b71;
			c22 += a20*b02 + a21*b12 + a22*b22 + a23*b32 + a24*b42 + a25*b52 + a26*b62 + a27*b72;
			c23 += a20*b03 + a21*b13 + a22*b23 + a23*b33 + a24*b43 + a25*b53 + a26*b63 + a27*b73;
			c24 += a20*b04 + a21*b14 + a22*b24 + a23*b34 + a24*b44 + a25*b54 + a26*b64 + a27*b74;
			c25 += a20*b05 + a21*b15 + a22*b25 + a23*b35 + a24*b45 + a25*b55 + a26*b65 + a27*b75;
			c26 += a20*b06 + a21*b16 + a22*b26 + a23*b36 + a24*b46 + a25*b56 + a26*b66 + a27*b76;
			c27 += a20*b07 + a21*b17 + a22*b27 + a23*b37 + a24*b47 + a25*b57 + a26*b67 + a27*b77;

			c30 += a30*b00 + a31*b10 + a32*b20 + a33*b30 + a34*b40 + a35*b50 + a36*b60 + a37*b70;
			c31 += a30*b01 + a31*b11 + a32*b21 + a33*b31 + a34*b41 + a35*b51 + a36*b61 + a37*b71;
			c32 += a30*b02 + a31*b12 + a32*b22 + a33*b32 + a34*b42 + a35*b52 + a36*b62 + a37*b72;
			c33 += a30*b03 + a31*b13 + a32*b23 + a33*b33 + a34*b43 + a35*b53 + a36*b63 + a37*b73;
			c34 += a30*b04 + a31*b14 + a32*b24 + a33*b34 + a34*b44 + a35*b54 + a36*b64 + a37*b74;
			c35 += a30*b05 + a31*b15 + a32*b25 + a33*b35 + a34*b45 + a35*b55 + a36*b65 + a37*b75;
			c36 += a30*b06 + a31*b16 + a32*b26 + a33*b36 + a34*b46 + a35*b56 + a36*b66 + a37*b76;
			c37 += a30*b07 + a31*b17 + a32*b27 + a33*b37 + a34*b47 + a35*b57 + a36*b67 + a37*b77;

			c40 += a40*b00 + a41*b10 + a42*b20 + a43*b30 + a44*b40 + a45*b50 + a46*b60 + a47*b70;
			c41 += a40*b01 + a41*b11 + a42*b21 + a43*b31 + a44*b41 + a45*b51 + a46*b61 + a47*b71;
			c42 += a40*b02 + a41*b12 + a42*b22 + a43*b32 + a44*b42 + a45*b52 + a46*b62 + a47*b72;
			c43 += a40*b03 + a41*b13 + a42*b23 + a43*b33 + a44*b43 + a45*b53 + a46*b63 + a47*b73;
			c44 += a40*b04 + a41*b14 + a42*b24 + a43*b34 + a44*b44 + a45*b54 + a46*b64 + a47*b74;
			c45 += a40*b05 + a41*b15 + a42*b25 + a43*b35 + a44*b45 + a45*b55 + a46*b65 + a47*b75;
			c46 += a40*b06 + a41*b16 + a42*b26 + a43*b36 + a44*b46 + a45*b56 + a46*b66 + a47*b76;
			c47 += a40*b07 + a41*b17 + a42*b27 + a43*b37 + a44*b47 + a45*b57 + a46*b67 + a47*b77;

			c50 += a50*b00 + a51*b10 + a52*b20 + a53*b30 + a54*b40 + a55*b50 + a56*b60 + a57*b70;
			c51 += a50*b01 + a51*b11 + a52*b21 + a53*b31 + a54*b41 + a55*b51 + a56*b61 + a57*b71;
			c52 += a50*b02 + a51*b12 + a52*b22 + a53*b32 + a54*b42 + a55*b52 + a56*b62 + a57*b72;
			c53 += a50*b03 + a51*b13 + a52*b23 + a53*b33 + a54*b43 + a55*b53 + a56*b63 + a57*b73;
			c54 += a50*b04 + a51*b14 + a52*b24 + a53*b34 + a54*b44 + a55*b54 + a56*b64 + a57*b74;
			c55 += a50*b05 + a51*b15 + a52*b25 + a53*b35 + a54*b45 + a55*b55 + a56*b65 + a57*b75;
			c56 += a50*b06 + a51*b16 + a52*b26 + a53*b36 + a54*b46 + a55*b56 + a56*b66 + a57*b76;
			c57 += a50*b07 + a51*b17 + a52*b27 + a53*b37 + a54*b47 + a55*b57 + a56*b67 + a57*b77;

			c60 += a60*b00 + a61*b10 + a62*b20 + a63*b30 + a64*b40 + a65*b50 + a66*b60 + a67*b70;
			c61 += a60*b01 + a61*b11 + a62*b21 + a63*b31 + a64*b41 + a65*b51 + a66*b61 + a67*b71;
			c62 += a60*b02 + a61*b12 + a62*b22 + a63*b32 + a64*b42 + a65*b52 + a66*b62 + a67*b72;
			c63 += a60*b03 + a61*b13 + a62*b23 + a63*b33 + a64*b43 + a65*b53 + a66*b63 + a67*b73;
			c64 += a60*b04 + a61*b14 + a62*b24 + a63*b34 + a64*b44 + a65*b54 + a66*b64 + a67*b74;
			c65 += a60*b05 + a61*b15 + a62*b25 + a63*b35 + a64*b45 + a65*b55 + a66*b65 + a67*b75;
			c66 += a60*b06 + a61*b16 + a62*b26 + a63*b36 + a64*b46 + a65*b56 + a66*b66 + a67*b76;
			c67 += a60*b07 + a61*b17 + a62*b27 + a63*b37 + a64*b47 + a65*b57 + a66*b67 + a67*b77;

			c70 += a70*b00 + a71*b10 + a72*b20 + a73*b30 + a74*b40 + a75*b50 + a76*b60 + a77*b70;
			c71 += a70*b01 + a71*b11 + a72*b21 + a73*b31 + a74*b41 + a75*b51 + a76*b61 + a77*b71;
			c72 += a70*b02 + a71*b12 + a72*b22 + a73*b32 + a74*b42 + a75*b52 + a76*b62 + a77*b72;
			c73 += a70*b03 + a71*b13 + a72*b23 + a73*b33 + a74*b43 + a75*b53 + a76*b63 + a77*b73;
			c74 += a70*b04 + a71*b14 + a72*b24 + a73*b34 + a74*b44 + a75*b54 + a76*b64 + a77*b74;
			c75 += a70*b05 + a71*b15 + a72*b25 + a73*b35 + a74*b45 + a75*b55 + a76*b65 + a77*b75;
			c76 += a70*b06 + a71*b16 + a72*b26 + a73*b36 + a74*b46 + a75*b56 + a76*b66 + a77*b76;
			c77 += a70*b07 + a71*b17 + a72*b27 + a73*b37 + a74*b47 + a75*b57 + a76*b67 + a77*b77;


			++ii;
		}
		__syncthreads();
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
