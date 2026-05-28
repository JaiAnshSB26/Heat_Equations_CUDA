#include "heat2d.hpp"
#include <iostream>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

void step_once_cuda(const double* d_cur, double* d_nxt, const HeatParams& p);

int main() {
    HeatParams p;
    p.width = 2048;
    p.height = 2048;
    p.lambda = 0.24;
    p.num_steps = 2000;
    p.snapshot_every = 2000;
    p.out_dir = "snapshots_cuda";

    Grid h_cur = make_grid(p);
    Grid h_nxt(p.width * p.height, 0.0);

    size_t grid_bytes = p.width * p.height * sizeof(double);

    double *d_cur = nullptr;
    double *d_nxt = nullptr;
    cudaMalloc(&d_cur, grid_bytes);
    cudaMalloc(&d_nxt, grid_bytes);

    cudaMemcpy(d_cur, h_cur.data(), grid_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nxt, h_nxt.data(), grid_bytes, cudaMemcpyHostToDevice);

    std::cout << "Starting CUDA heat simulation loop...\n";
    const auto start = std::chrono::high_resolution_clock::now();

    for (int s = 1; s <= p.num_steps; ++s) {
        step_once_cuda(d_cur, d_nxt, p);
        
        std::swap(d_cur, d_nxt);
    }

    cudaDeviceSynchronize();

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(h_cur.data(), d_cur, grid_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Total GPU Engine Compute Runtime: " << elapsed.count() << " seconds\n";

    write_snapshot(h_cur, p, p.num_steps);

    cudaFree(d_cur);
    cudaFree(d_nxt);

    return 0;
}