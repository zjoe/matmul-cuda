#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "tiled_matmul-9-3.fatbin.c"
extern void __device_stub__Z6matMulPKfS0_Pfiii(const float *, const float *, float *, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_24_tiled_matmul_9_3_cpp1_ii_tex_A(void) __attribute__((__constructor__));
void __device_stub__Z6matMulPKfS0_Pfiii(const float *__par0, const float *__par1, float *__par2, int __par3, int __par4, int __par5){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, float *, int, int, int))matMul)));}
# 18 "tiled_matmul-9-3.cu"
void matMul( const float *__cuda_0,const float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 19 "tiled_matmul-9-3.cu"
{__device_stub__Z6matMulPKfS0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 205 "tiled_matmul-9-3.cu"
}
# 1 "tiled_matmul-9-3.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T242) {  __nv_dummy_param_ref(__T242); __nv_save_fatbinhandle_for_managed_rt(__T242); __cudaRegisterEntry(__T242, ((void ( *)(const float *, const float *, float *, int, int, int))matMul), _Z6matMulPKfS0_Pfiii, (-1)); __cudaRegisterGlobalTexture(__T242, __text_var(tex_A,::tex_A), 2, 0, 0); __cudaRegisterGlobalTexture(__T242, __text_var(tex_B,::tex_B), 2, 0, 0); __cudaRegisterGlobalSurface(__T242, __text_var(surf_C,::surf_C), 2, 0); }
static void __sti____cudaRegisterAll_24_tiled_matmul_9_3_cpp1_ii_tex_A(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
