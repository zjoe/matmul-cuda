# 1 "tiled_matmul-9-3.cudafe1.gpu"
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 6 "tiled_matmul-9-3.cu"
__texture_type__ __text_var(tex_A,::tex_A);
# 7 "tiled_matmul-9-3.cu"
__texture_type__ __text_var(tex_B,::tex_B);
# 8 "tiled_matmul-9-3.cu"
__surface_type__ __text_var(surf_C,::surf_C);

#include "tiled_matmul-9-3.cudafe2.stub.c"
