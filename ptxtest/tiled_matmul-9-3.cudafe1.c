# 1 "tiled_matmul-9-3.cu"
# 56 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
#pragma GCC diagnostic push


#pragma GCC diagnostic ignored "-Wunused-function"
# 35 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility push ( default )
# 149 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility pop
# 42 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility push ( default )
# 120 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility pop
# 1888 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 764 "/usr/local/cuda/bin/..//include/driver_types.h"
struct cudaArray;
# 1425 "/usr/local/cuda/bin/..//include/driver_types.h"
struct CUstream_st;




struct CUevent_st;
# 180 "/usr/include/libio.h" 3
enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv};
# 245 "/usr/include/libio.h" 3
struct _IO_FILE;
# 51 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
enum idtype_t {
P_ALL,
P_PID,
P_PGID};
# 190 "/usr/include/math.h" 3
enum _ZUt_ {
FP_NAN,


FP_INFINITE,


FP_ZERO,


FP_SUBNORMAL,


FP_NORMAL};
# 302 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_};
# 78 "/usr/local/cuda/bin/..//include/cuda_surface_types.h"
struct _Z7surfaceIvLi2EE;
# 78 "/usr/local/cuda/bin/..//include/cuda_texture_types.h"
struct _Z7textureIfLi2EL19cudaTextureReadMode0EE;
# 128 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E { _ZNSt9__is_voidIvE7__valueE = 1};
# 148 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E { _ZNSt12__is_integerIbE7__valueE = 1};
# 155 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E { _ZNSt12__is_integerIcE7__valueE = 1};
# 162 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E { _ZNSt12__is_integerIaE7__valueE = 1};
# 169 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E { _ZNSt12__is_integerIhE7__valueE = 1};
# 177 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E { _ZNSt12__is_integerIwE7__valueE = 1};
# 201 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E { _ZNSt12__is_integerIsE7__valueE = 1};
# 208 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E { _ZNSt12__is_integerItE7__valueE = 1};
# 215 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E { _ZNSt12__is_integerIiE7__valueE = 1};
# 222 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E { _ZNSt12__is_integerIjE7__valueE = 1};
# 229 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E { _ZNSt12__is_integerIlE7__valueE = 1};
# 236 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E { _ZNSt12__is_integerImE7__valueE = 1};
# 243 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E { _ZNSt12__is_integerIxE7__valueE = 1};
# 250 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E { _ZNSt12__is_integerIyE7__valueE = 1};
# 268 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E { _ZNSt13__is_floatingIfE7__valueE = 1};
# 275 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E { _ZNSt13__is_floatingIdE7__valueE = 1};
# 282 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E { _ZNSt13__is_floatingIeE7__valueE = 1};
# 358 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E { _ZNSt9__is_charIcE7__valueE = 1};
# 366 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E { _ZNSt9__is_charIwE7__valueE = 1};
# 381 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E { _ZNSt9__is_byteIcE7__valueE = 1};
# 388 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E { _ZNSt9__is_byteIaE7__valueE = 1};
# 395 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E { _ZNSt9__is_byteIhE7__valueE = 1};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIeEUt_E { _ZNSt12__is_integerIeE7__valueE}; enum _ZNSt12__is_integerIdEUt_E { _ZNSt12__is_integerIdE7__valueE}; enum _ZNSt12__is_integerIfEUt_E { _ZNSt12__is_integerIfE7__valueE};
# 153 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_;
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
# 764 "/usr/local/cuda/bin/..//include/driver_types.h"
typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;
# 48 "/usr/include/stdio.h" 3
typedef struct _IO_FILE FILE;
# 78 "/usr/local/cuda/bin/..//include/cuda_surface_types.h"
struct _Z7surfaceIvLi2EE { struct surfaceReference __b_16surfaceReference;};
# 78 "/usr/local/cuda/bin/..//include/cuda_texture_types.h"
struct _Z7textureIfLi2EL19cudaTextureReadMode0EE { struct textureReference __b_16textureReference;};
# 153 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_ { long double __l; int __i[3];};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 251 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaDeviceReset(void);
# 2193 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaEventCreate(struct CUevent_st **);
# 2322 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaEventSynchronize(struct CUevent_st *);
# 2388 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaEventElapsedTime(float *, struct CUevent_st *, struct CUevent_st *);
# 2782 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaConfigureCall(struct dim3, struct dim3, size_t, struct CUstream_st *);
# 3074 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMallocArray(cudaArray_t *, const struct cudaChannelFormatDesc *, size_t, size_t, unsigned);
# 3999 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
# 4065 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMemcpyToArray(cudaArray_t, size_t, size_t, const void *, size_t, enum cudaMemcpyKind);
# 5572 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaGetChannelDesc(struct cudaChannelFormatDesc *, cudaArray_const_t);
# 5607 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern struct cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, enum cudaChannelFormatKind);
# 5733 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaBindTextureToArray(const struct textureReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
# 5872 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaBindSurfaceToArray(const struct surfaceReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
# 237 "/usr/include/stdio.h" 3
extern int fclose(FILE *);
# 272 "/usr/include/stdio.h" 3
extern FILE *fopen(const char *__restrict__, const char *__restrict__);
# 85 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 3
extern int __fprintf_chk(FILE *__restrict__, int, const char *__restrict__, ...);

extern int __printf_chk(int, const char *__restrict__, ...);
# 183 "/usr/include/stdlib.h" 3
extern __attribute__((__nothrow__)) long strtol(const char *__restrict__, char **__restrict__, int);
# 374 "/usr/include/stdlib.h" 3
extern __attribute__((__nothrow__)) int rand(void);
# 543 "/usr/include/stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void exit(int);
# 10 "tiled_matmul-9-3.cu"
extern void _Z12err_handlingP9cudaErrorPKc(enum cudaError *, const char *);
# 206 "tiled_matmul-9-3.cu"
extern int main(int, char **);
# 1057 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static __inline__ enum cudaError _Z22cudaBindTextureToArrayIfLi2EL19cudaTextureReadMode0EE9cudaErrorRK7textureIT_XT0_EXT1_EEPK9cudaArray(const struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *, cudaArray_const_t);
# 1868 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static __inline__ enum cudaError _Z22cudaBindSurfaceToArrayIvLi2EE9cudaErrorRK7surfaceIT_XT0_EEPK9cudaArray(const struct _Z7surfaceIvLi2EE *, cudaArray_const_t);
extern int __cudaSetupArgSimple();
extern int __cudaLaunch();
# 97 "/usr/local/cuda/bin/..//include/cuda_surface_types.h"
extern  __attribute__((__weak__)) /* COMDAT group: _ZN7surfaceIvLi2EEC1Ev */ __inline__ void _ZN7surfaceIvLi2EEC1Ev(struct _Z7surfaceIvLi2EE *const);
extern  __attribute__((__weak__)) /* COMDAT group: _ZN7surfaceIvLi2EEC2Ev */ __inline__ void _ZN7surfaceIvLi2EEC2Ev(struct _Z7surfaceIvLi2EE *const);
# 81 "/usr/local/cuda/bin/..//include/cuda_texture_types.h"
extern  __attribute__((__weak__)) /* COMDAT group: _ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode */ __inline__ void _ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode(struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *const, int, enum cudaTextureFilterMode, enum cudaTextureAddressMode);
extern  __attribute__((__weak__)) /* COMDAT group: _ZN7textureIfLi2EL19cudaTextureReadMode0EEC2Ei21cudaTextureFilterMode22cudaTextureAddressMode */ __inline__ void _ZN7textureIfLi2EL19cudaTextureReadMode0EEC2Ei21cudaTextureFilterMode22cudaTextureAddressMode(struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *const, int, enum cudaTextureFilterMode, enum cudaTextureAddressMode);
extern void __nv_dummy_param_ref();
extern void __nv_save_fatbinhandle_for_managed_rt();
extern int __cudaRegisterEntry();
extern int __cudaRegisterGlobalTexture();
extern int __cudaRegisterGlobalSurface();
extern int __cudaRegisterBinary();
static void __sti___24_tiled_matmul_9_3_cpp1_ii_tex_A(void) __attribute__((__constructor__));
# 6 "tiled_matmul-9-3.cu"
struct _Z7textureIfLi2EL19cudaTextureReadMode0EE __text_var(tex_A,::tex_A);
struct _Z7textureIfLi2EL19cudaTextureReadMode0EE __text_var(tex_B,::tex_B);
struct _Z7surfaceIvLi2EE __text_var(surf_C,::surf_C);

void _Z12err_handlingP9cudaErrorPKc( enum cudaError *err,  const char *str)
{
if (((int)(*err)) != 0) {
printf(((const char *)"%s\n"), str);
exit(1);
} 
}
# 206 "tiled_matmul-9-3.cu"
int main( int argc,  char **argv)
{  const char *__T20;
 const char *__T21;
 const char *__T22;
 int __T23;
 int __T24;
 int __T25;
 unsigned __T26;
 unsigned __T27;
# 213 "tiled_matmul-9-3.cu"
 enum cudaError __cuda_local_var_43735_14_non_const_err;

 int __cuda_local_var_43737_6_non_const_m;
 int __cuda_local_var_43738_6_non_const_n;
 int __cuda_local_var_43739_6_non_const_k;

 float *__cuda_local_var_43741_9_non_const_A;
 float *__cuda_local_var_43742_9_non_const_B;
 float *__cuda_local_var_43743_9_non_const_C;
# 237 "tiled_matmul-9-3.cu"
 float *__cuda_local_var_43759_9_non_const_dev_A;
 float *__cuda_local_var_43760_9_non_const_dev_B;
 float *__cuda_local_var_43761_9_non_const_dev_C;
# 256 "tiled_matmul-9-3.cu"
 struct cudaChannelFormatDesc __cuda_local_var_43778_24_non_const_ADesc;
 struct cudaChannelFormatDesc __cuda_local_var_43779_24_non_const_BDesc;
 struct cudaChannelFormatDesc __cuda_local_var_43780_24_non_const_CDesc;
 struct cudaArray *__cuda_local_var_43781_13_non_const_A_array;
# 259 "tiled_matmul-9-3.cu"
 struct cudaArray *__cuda_local_var_43781_23_non_const_B_array;
# 259 "tiled_matmul-9-3.cu"
 struct cudaArray *__cuda_local_var_43781_33_non_const_C_array;
# 278 "tiled_matmul-9-3.cu"
 struct dim3 __cuda_local_var_43800_7_non_const_dimGrid;
 struct dim3 __cuda_local_var_43801_7_non_const_dimBlock;

 struct CUevent_st *__cuda_local_var_43803_14_non_const_start;
# 281 "tiled_matmul-9-3.cu"
 struct CUevent_st *__cuda_local_var_43803_21_non_const_stop;
# 293 "tiled_matmul-9-3.cu"
 float __cuda_local_var_43815_8_non_const_time_elapsed;
# 302 "tiled_matmul-9-3.cu"
 FILE *__cuda_local_var_43824_8_non_const_fp;
# 208 "tiled_matmul-9-3.cu"
if (argc != 4) {
printf(((const char *)"usage: ./xxx m n k\n"));
return (-1);
}

__cuda_local_var_43735_14_non_const_err = ((enum cudaError)0);

__cuda_local_var_43737_6_non_const_m = ((__T20 = ((const char *)(argv[1]))) , ((int)(strtol(__T20, ((char **)0LL), 10))));
__cuda_local_var_43738_6_non_const_n = ((__T21 = ((const char *)(argv[2]))) , ((int)(strtol(__T21, ((char **)0LL), 10))));
__cuda_local_var_43739_6_non_const_k = ((__T22 = ((const char *)(argv[3]))) , ((int)(strtol(__T22, ((char **)0LL), 10))));

__cuda_local_var_43741_9_non_const_A = ((float *)(malloc((((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43739_6_non_const_k)) * 4UL))));
__cuda_local_var_43742_9_non_const_B = ((float *)(malloc((((unsigned long)(__cuda_local_var_43739_6_non_const_k * __cuda_local_var_43738_6_non_const_n)) * 4UL))));
__cuda_local_var_43743_9_non_const_C = ((float *)(malloc((((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43738_6_non_const_n)) * 4UL))));

if (((__cuda_local_var_43741_9_non_const_A == ((float *)0LL)) || (__cuda_local_var_43742_9_non_const_B == ((float *)0LL))) || (__cuda_local_var_43743_9_non_const_C == ((float *)0LL))) {
printf(((const char *)"allocate host error!\n"));
return 1;
} {

 int i;
# 228 "tiled_matmul-9-3.cu"
i = 0; for (; (i < (__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43739_6_non_const_k)); ++i) {
(__cuda_local_var_43741_9_non_const_A[i]) = ((((float)(rand())) / (2147483648.0F)) - (((float)(rand())) / (2147483648.0F)));
} } {

 int i;
# 232 "tiled_matmul-9-3.cu"
i = 0; for (; (i < (__cuda_local_var_43739_6_non_const_k * __cuda_local_var_43738_6_non_const_n)); ++i) {
(__cuda_local_var_43742_9_non_const_B[i]) = ((((float)(rand())) / (2147483648.0F)) - (((float)(rand())) / (2147483648.0F)));
} }


__cuda_local_var_43759_9_non_const_dev_A = ((float *)0LL);
__cuda_local_var_43760_9_non_const_dev_B = ((float *)0LL);
__cuda_local_var_43761_9_non_const_dev_C = ((float *)0LL);

__cuda_local_var_43735_14_non_const_err = (cudaMalloc(((void **)(&__cuda_local_var_43759_9_non_const_dev_A)), (((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43739_6_non_const_k)) * 4UL)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"allocate devecie error A!"));

__cuda_local_var_43735_14_non_const_err = (cudaMalloc(((void **)(&__cuda_local_var_43760_9_non_const_dev_B)), (((unsigned long)(__cuda_local_var_43739_6_non_const_k * __cuda_local_var_43738_6_non_const_n)) * 4UL)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"allocate devecie error B!"));

__cuda_local_var_43735_14_non_const_err = (cudaMalloc(((void **)(&__cuda_local_var_43761_9_non_const_dev_C)), (((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43738_6_non_const_n)) * 4UL)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"allocate devecie error C!"));

__cuda_local_var_43735_14_non_const_err = (cudaMemcpy(((void *)__cuda_local_var_43759_9_non_const_dev_A), ((const void *)__cuda_local_var_43741_9_non_const_A), (((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43739_6_non_const_k)) * 4UL), cudaMemcpyHostToDevice));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"memcpy to A error!"));

__cuda_local_var_43735_14_non_const_err = (cudaMemcpy(((void *)__cuda_local_var_43760_9_non_const_dev_B), ((const void *)__cuda_local_var_43742_9_non_const_B), (((unsigned long)(__cuda_local_var_43739_6_non_const_k * __cuda_local_var_43738_6_non_const_n)) * 4UL), cudaMemcpyHostToDevice));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"memcpy to B error!"));

__cuda_local_var_43778_24_non_const_ADesc = ((__T23 = 32) , (cudaCreateChannelDesc(__T23, 0, 0, 0, cudaChannelFormatKindFloat)));
__cuda_local_var_43779_24_non_const_BDesc = ((__T24 = 32) , (cudaCreateChannelDesc(__T24, 0, 0, 0, cudaChannelFormatKindFloat)));
__cuda_local_var_43780_24_non_const_CDesc = ((__T25 = 32) , (cudaCreateChannelDesc(__T25, 0, 0, 0, cudaChannelFormatKindFloat)));

cudaMallocArray((&__cuda_local_var_43781_13_non_const_A_array), ((const struct cudaChannelFormatDesc *)(&__cuda_local_var_43778_24_non_const_ADesc)), ((size_t)__cuda_local_var_43739_6_non_const_k), ((size_t)__cuda_local_var_43737_6_non_const_m), 0U);
cudaMallocArray((&__cuda_local_var_43781_23_non_const_B_array), ((const struct cudaChannelFormatDesc *)(&__cuda_local_var_43779_24_non_const_BDesc)), ((size_t)__cuda_local_var_43738_6_non_const_n), ((size_t)__cuda_local_var_43739_6_non_const_k), 0U);
cudaMallocArray((&__cuda_local_var_43781_33_non_const_C_array), ((const struct cudaChannelFormatDesc *)(&__cuda_local_var_43780_24_non_const_CDesc)), ((size_t)__cuda_local_var_43738_6_non_const_n), ((size_t)__cuda_local_var_43737_6_non_const_m), 2U);
cudaMemcpyToArray(__cuda_local_var_43781_13_non_const_A_array, 0UL, 0UL, ((const void *)__cuda_local_var_43741_9_non_const_A), (((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43739_6_non_const_k)) * 4UL), cudaMemcpyHostToDevice);
cudaMemcpyToArray(__cuda_local_var_43781_23_non_const_B_array, 0UL, 0UL, ((const void *)__cuda_local_var_43742_9_non_const_B), (((unsigned long)(__cuda_local_var_43739_6_non_const_k * __cuda_local_var_43738_6_non_const_n)) * 4UL), cudaMemcpyHostToDevice);

_Z22cudaBindTextureToArrayIfLi2EL19cudaTextureReadMode0EE9cudaErrorRK7textureIT_XT0_EXT1_EEPK9cudaArray((((const struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *)&__text_var(tex_A,::tex_A))), ((cudaArray_const_t)__cuda_local_var_43781_13_non_const_A_array));
_Z22cudaBindTextureToArrayIfLi2EL19cudaTextureReadMode0EE9cudaErrorRK7textureIT_XT0_EXT1_EEPK9cudaArray((((const struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *)&__text_var(tex_B,::tex_B))), ((cudaArray_const_t)__cuda_local_var_43781_23_non_const_B_array));
_Z22cudaBindSurfaceToArrayIvLi2EE9cudaErrorRK7surfaceIT_XT0_EEPK9cudaArray((((const struct _Z7surfaceIvLi2EE *)&__text_var(surf_C,::surf_C))), ((cudaArray_const_t)__cuda_local_var_43781_33_non_const_C_array));

((((__text_var(tex_A,::tex_A).__b_16textureReference).addressMode))[0]) = cudaAddressModeBorder;
((((__text_var(tex_A,::tex_A).__b_16textureReference).addressMode))[1]) = cudaAddressModeBorder;

((((__text_var(tex_B,::tex_B).__b_16textureReference).addressMode))[0]) = cudaAddressModeBorder;
((((__text_var(tex_B,::tex_B).__b_16textureReference).addressMode))[1]) = cudaAddressModeBorder;



{ __T26 = ((unsigned)(((__cuda_local_var_43738_6_non_const_n - 1) / 64) + 1)); __T27 = ((unsigned)(((__cuda_local_var_43737_6_non_const_m - 1) / 64) + 1));
# 421 "/usr/local/cuda/bin/..//include/vector_types.h"
{ (__cuda_local_var_43800_7_non_const_dimGrid.x) = __T26; (__cuda_local_var_43800_7_non_const_dimGrid.y) = __T27; (__cuda_local_var_43800_7_non_const_dimGrid.z) = 1U; }
# 278 "tiled_matmul-9-3.cu"
}
{
# 421 "/usr/local/cuda/bin/..//include/vector_types.h"
(__cuda_local_var_43801_7_non_const_dimBlock.x) = 8U; (__cuda_local_var_43801_7_non_const_dimBlock.y) = 8U; (__cuda_local_var_43801_7_non_const_dimBlock.z) = 1U;
# 279 "tiled_matmul-9-3.cu"
}



cudaEventCreate((&__cuda_local_var_43803_14_non_const_start));
cudaEventCreate((&__cuda_local_var_43803_21_non_const_stop));

cudaEventRecord(__cuda_local_var_43803_14_non_const_start, ((struct CUstream_st *)0LL));
(cudaConfigureCall(__cuda_local_var_43800_7_non_const_dimGrid, __cuda_local_var_43801_7_non_const_dimBlock, 0UL, ((struct CUstream_st *)0LL))) ? ((void)0) : (__device_stub__Z6matMulPKfS0_Pfiii(((const float *)__cuda_local_var_43759_9_non_const_dev_A), ((const float *)__cuda_local_var_43760_9_non_const_dev_B), __cuda_local_var_43761_9_non_const_dev_C, __cuda_local_var_43737_6_non_const_m, __cuda_local_var_43739_6_non_const_k, __cuda_local_var_43738_6_non_const_n));
cudaEventRecord(__cuda_local_var_43803_21_non_const_stop, ((struct CUstream_st *)0LL));

cudaEventSynchronize(__cuda_local_var_43803_14_non_const_start);
cudaEventSynchronize(__cuda_local_var_43803_21_non_const_stop);

__cuda_local_var_43815_8_non_const_time_elapsed = (0.0F);
cudaEventElapsedTime((&__cuda_local_var_43815_8_non_const_time_elapsed), __cuda_local_var_43803_14_non_const_start, __cuda_local_var_43803_21_non_const_stop);
printf(((const char *)"%fms\n"), ((double)__cuda_local_var_43815_8_non_const_time_elapsed));


__cuda_local_var_43735_14_non_const_err = (cudaMemcpy(((void *)__cuda_local_var_43743_9_non_const_C), ((const void *)__cuda_local_var_43761_9_non_const_dev_C), (((unsigned long)(__cuda_local_var_43737_6_non_const_m * __cuda_local_var_43738_6_non_const_n)) * 4UL), cudaMemcpyDeviceToHost));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"memcpy to host C error!"));


__cuda_local_var_43824_8_non_const_fp = (fopen(((const char *)"gpu.out"), ((const char *)"w"))); {
 int i;
# 303 "tiled_matmul-9-3.cu"
i = 0; for (; (i < __cuda_local_var_43737_6_non_const_m); i++) { {
 int j;
# 304 "tiled_matmul-9-3.cu"
j = 0; for (; (j < __cuda_local_var_43738_6_non_const_n); j++) {
fprintf(__cuda_local_var_43824_8_non_const_fp, ((const char *)"%f\n"), ((double)(__cuda_local_var_43743_9_non_const_C[((i * __cuda_local_var_43738_6_non_const_n) + j)])));
} }
} }
fclose(__cuda_local_var_43824_8_non_const_fp);

__cuda_local_var_43735_14_non_const_err = (cudaFree(((void *)__cuda_local_var_43759_9_non_const_dev_A)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"mem free A error!"));

__cuda_local_var_43735_14_non_const_err = (cudaFree(((void *)__cuda_local_var_43760_9_non_const_dev_B)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"mem free B error!"));

__cuda_local_var_43735_14_non_const_err = (cudaFree(((void *)__cuda_local_var_43761_9_non_const_dev_C)));
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"mem free C error!"));

__cuda_local_var_43735_14_non_const_err = (cudaDeviceReset());
_Z12err_handlingP9cudaErrorPKc((&__cuda_local_var_43735_14_non_const_err), ((const char *)"device reset error!"));

return 0;
}
# 1057 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static __inline__ enum cudaError _Z22cudaBindTextureToArrayIfLi2EL19cudaTextureReadMode0EE9cudaErrorRK7textureIT_XT0_EXT1_EEPK9cudaArray(
const struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *tex, 
cudaArray_const_t array)

{
 struct cudaChannelFormatDesc __cuda_local_var_43214_32_non_const_desc;
 enum cudaError __cuda_local_var_43215_15_non_const_err;
# 1063 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
__cuda_local_var_43215_15_non_const_err = (cudaGetChannelDesc((&__cuda_local_var_43214_32_non_const_desc), array));

return (((int)__cuda_local_var_43215_15_non_const_err) == 0) ? (cudaBindTextureToArray((&(tex->__b_16textureReference)), array, (((const struct cudaChannelFormatDesc *)&__cuda_local_var_43214_32_non_const_desc)))) : __cuda_local_var_43215_15_non_const_err;
}
# 1868 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static __inline__ enum cudaError _Z22cudaBindSurfaceToArrayIvLi2EE9cudaErrorRK7surfaceIT_XT0_EEPK9cudaArray(
const struct _Z7surfaceIvLi2EE *surf, 
cudaArray_const_t array)

{
 struct cudaChannelFormatDesc __cuda_local_var_43548_32_non_const_desc;
 enum cudaError __cuda_local_var_43549_15_non_const_err;
# 1874 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
__cuda_local_var_43549_15_non_const_err = (cudaGetChannelDesc((&__cuda_local_var_43548_32_non_const_desc), array));

return (((int)__cuda_local_var_43549_15_non_const_err) == 0) ? (cudaBindSurfaceToArray((&(surf->__b_16surfaceReference)), array, (((const struct cudaChannelFormatDesc *)&__cuda_local_var_43548_32_non_const_desc)))) : __cuda_local_var_43549_15_non_const_err;
}
__asm__(".align 2");
# 97 "/usr/local/cuda/bin/..//include/cuda_surface_types.h"
 __attribute__((__weak__)) /* COMDAT group: _ZN7surfaceIvLi2EEC1Ev */ __inline__ void _ZN7surfaceIvLi2EEC1Ev( struct _Z7surfaceIvLi2EE *const this)
{
((this->__b_16surfaceReference).channelDesc) = (cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone)); 
}
__asm__(".align 2");
 __attribute__((__weak__)) /* COMDAT group: _ZN7surfaceIvLi2EEC2Ev */ __inline__ void _ZN7surfaceIvLi2EEC2Ev( struct _Z7surfaceIvLi2EE *const this) {  _ZN7surfaceIvLi2EEC1Ev(this);  }
__asm__(".align 2");
# 81 "/usr/local/cuda/bin/..//include/cuda_texture_types.h"
 __attribute__((__weak__)) /* COMDAT group: _ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode */ __inline__ void _ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode( struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *const this,  int norm, 
enum cudaTextureFilterMode fMode, 
enum cudaTextureAddressMode aMode)
{  int __T238;
((this->__b_16textureReference).normalized) = norm;
((this->__b_16textureReference).filterMode) = fMode;
((((this->__b_16textureReference).addressMode))[0]) = aMode;
((((this->__b_16textureReference).addressMode))[1]) = aMode;
((((this->__b_16textureReference).addressMode))[2]) = aMode;
((this->__b_16textureReference).channelDesc) = ((__T238 = 32) , (cudaCreateChannelDesc(__T238, 0, 0, 0, cudaChannelFormatKindFloat)));
((this->__b_16textureReference).sRGB) = 0; 
}
__asm__(".align 2");
 __attribute__((__weak__)) /* COMDAT group: _ZN7textureIfLi2EL19cudaTextureReadMode0EEC2Ei21cudaTextureFilterMode22cudaTextureAddressMode */ __inline__ void _ZN7textureIfLi2EL19cudaTextureReadMode0EEC2Ei21cudaTextureFilterMode22cudaTextureAddressMode( struct _Z7textureIfLi2EL19cudaTextureReadMode0EE *const this,  int __T239,  enum cudaTextureFilterMode __T240,  enum cudaTextureAddressMode __T241) {  _ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode(this, __T239, __T240, __T241);  }
static void __sti___24_tiled_matmul_9_3_cpp1_ii_tex_A(void) {
# 6 "tiled_matmul-9-3.cu"
_ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode((&__text_var(tex_A,::tex_A)), 0, cudaFilterModePoint, cudaAddressModeClamp);
_ZN7textureIfLi2EL19cudaTextureReadMode0EEC1Ei21cudaTextureFilterMode22cudaTextureAddressMode((&__text_var(tex_B,::tex_B)), 0, cudaFilterModePoint, cudaAddressModeClamp);
_ZN7surfaceIvLi2EEC1Ev((&__text_var(surf_C,::surf_C)));  }

#include "tiled_matmul-9-3.cudafe1.stub.c"
