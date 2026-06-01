#include "heat2d_cuda.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Abort on the first CUDA error; keeping call sites readable.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 16x16 = 256 threads/block: occupancy-friendly default on all recent GPUs.
static constexpr int kBlock = 16;

// One thread per cell. Thread (x, y) -> grid column j, row i (row-major).
// Interior-only update; boundary threads return so edges stay fixed at 0.
__global__ void heat_kernel(const double* cur, double* nxt,
                            int W, int H, double lam) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // column (x)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // row (y)

    if (i < 1 || i >= H - 1 || j < 1 || j >= W - 1) return;

    const int c = i * W + j;
    nxt[c] = (1.0 - 4.0 * lam) * cur[c]
           + lam * (cur[c - W] + cur[c + W] + cur[c - 1] + cur[c + 1]);
}

void step_once_cuda(const Grid& cur, Grid& nxt, const HeatParams& p) {
    const int W = p.width;
    const int H = p.height;
    const size_t bytes = static_cast<size_t>(W) * H * sizeof(double);

    double* d_cur = nullptr;
    double* d_nxt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, bytes));
    CUDA_CHECK(cudaMalloc(&d_nxt, bytes));

    CUDA_CHECK(cudaMemcpy(d_cur, cur.data(), bytes, cudaMemcpyHostToDevice));
    // Zero d_nxt so the untouched boundary ring stays at 0.
    CUDA_CHECK(cudaMemset(d_nxt, 0, bytes));

    const dim3 block(kBlock, kBlock);
    const dim3 grid((W + kBlock - 1) / kBlock, (H + kBlock - 1) / kBlock);

    heat_kernel<<<grid, block>>>(d_cur, d_nxt, W, H, p.lambda);
    CUDA_CHECK(cudaGetLastError());       // launch-time errors
    CUDA_CHECK(cudaDeviceSynchronize());  // execution-time errors

    CUDA_CHECK(cudaMemcpy(nxt.data(), d_nxt, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_nxt));
}