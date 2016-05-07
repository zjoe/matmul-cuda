0 2x2 tiled
1 separate shared memory to eliminate bank conflict
2
3 4x4 tiled
4 texture & surface to handle bound conditions
5 8x8 tiled
6 8x64 smem instead of 64x64, shfl
7 smem double buffer
8 register double buffer
9 direct shared mem load, double buffer both
10 float4 shared load 50%, double buffer shared
10-1 float4 shared load 100% double buffer shared
9-1 float4 shared load 100% double buffer both, shared double buffer fixed to scatter incalculating
9-2 float4 shared load 100% double both, register array: c(looped); shared double buffer fixed to scatter in calculating
9-3 change write back from surface to shared+global
9-4 bound conditions resolved
9-5 batched, compared with cublas batchedsgemm
