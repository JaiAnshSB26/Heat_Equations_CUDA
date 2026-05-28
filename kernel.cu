#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "heat2d.hpp"

__global__ void heat2d_kernel(const double* __restrict__ cur, 
                              double* __restrict__ nxt, 
                              int width, int height, 
                              double lam, double self) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        int idx_center = i * width + j;
        
        nxt[idx_center] = self * cur[idx_center]
                        + lam * cur[(i - 1) * width + j]
                        + lam * cur[(i + 1) * width + j]
                        + lam * cur[i * width + (j - 1)]
                        + lam * cur[i * width + (j + 1)];
    }
}

void step_once_cuda(const double* d_cur, double* d_nxt, const HeatParams& p) {
    const double lam = p.lambda;
    const double self = 1.0 - 4.0 * lam;

    dim3 blockSize(16, 16);

    dim3 gridSize((p.width + blockSize.x - 1) / blockSize.x,
                  (p.height + blockSize.y - 1) / blockSize.y);

    heat2d_kernel<<<gridSize, blockSize>>>(d_cur, d_nxt, p.width, p.height, lam, self);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}