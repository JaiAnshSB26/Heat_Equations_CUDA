#include "heat2d_implicit.cuh"

#include <cstdio>
#include <cstdlib>
#include <utility>  // std::swap
#include <cuda_runtime.h>

// Abort on the first CUDA error; keeps call sites readable.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static constexpr int kBlock = 16;

// One Jacobi sweep for the backward-Euler system:
//   u1[c] = (b[c] + lam*(neighbours of u0)) / (1 + 4*lam)
// b = U^n (fixed during the inner iteration); u0 = iterate k; u1 = iterate k+1.
// Interior-only; boundary threads return so the fixed-0 ring is preserved.
__global__ void jacobi_kernel(const double* b, const double* u0, double* u1,
                              int W, int H, double lam) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // column (x)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // row (y)

    if (i < 1 || i >= H - 1 || j < 1 || j >= W - 1) return;

    const int c = i * W + j;
    u1[c] = (b[c] + lam * (u0[c - W] + u0[c + W] + u0[c - 1] + u0[c + 1]))
          / (1.0 + 4.0 * lam);
}

void solve_implicit_cuda(const Grid& initial, Grid& out,
                         const HeatParams& p, int iters, float* gpu_ms) {
    const int W = p.width;
    const int H = p.height;
    const size_t bytes = static_cast<size_t>(W) * H * sizeof(double);
    const double lam = p.lambda;

    // d_b holds U^n (the RHS); d_u0/d_u1 are the ping-ponged Jacobi iterates.
    double* d_b = nullptr;
    double* d_u0 = nullptr;
    double* d_u1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_u0, bytes));
    CUDA_CHECK(cudaMalloc(&d_u1, bytes));

    CUDA_CHECK(cudaMemcpy(d_b, initial.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u0, initial.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u1, 0, bytes));  // boundary ring stays 0

    const dim3 block(kBlock, kBlock);
    const dim3 grid((W + kBlock - 1) / kBlock, (H + kBlock - 1) / kBlock);

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));

    for (int n = 0; n < p.num_steps; ++n) {
        // Solve A U^{n+1} = U^n by `iters` Jacobi sweeps. The warm start is the
        // current iterate already in d_u0 (= U^n for the first step).
        for (int k = 0; k < iters; ++k) {
            jacobi_kernel<<<grid, block>>>(d_b, d_u0, d_u1, W, H, lam);
            std::swap(d_u0, d_u1);  // d_u0 now holds the latest iterate
        }
        // d_u0 holds U^{n+1}; it becomes the RHS for the next step.
        CUDA_CHECK(cudaMemcpy(d_b, d_u0, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    if (gpu_ms) CUDA_CHECK(cudaEventElapsedTime(gpu_ms, beg, end));

    out.assign(static_cast<size_t>(W) * H, 0.0);
    CUDA_CHECK(cudaMemcpy(out.data(), d_b, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_u0));
    CUDA_CHECK(cudaFree(d_u1));
}

// CPU reference: identical scheme and iteration count, so the GPU result must
// match this to floating-point round-off.
void solve_implicit_cpu(const Grid& initial, Grid& out,
                        const HeatParams& p, int iters) {
    const int W = p.width;
    const int H = p.height;
    const double lam = p.lambda;
    const double denom = 1.0 + 4.0 * lam;

    Grid b = initial;            // U^n
    Grid u0 = initial;           // Jacobi iterate
    Grid u1(b.size(), 0.0);

    for (int n = 0; n < p.num_steps; ++n) {
        for (int k = 0; k < iters; ++k) {
            for (int i = 1; i < H - 1; ++i) {
                for (int j = 1; j < W - 1; ++j) {
                    const int c = idx(i, j, W);
                    u1[c] = (b[c] + lam * (u0[idx(i - 1, j, W)] + u0[idx(i + 1, j, W)]
                                          + u0[idx(i, j - 1, W)] + u0[idx(i, j + 1, W)]))
                            / denom;
                }
            }
            std::swap(u0, u1);
        }
        b = u0;  // U^{n+1}
    }
    out = b;
}
