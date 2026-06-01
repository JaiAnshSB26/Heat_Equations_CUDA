// CUDA driver: runs the CPU reference and the GPU solver on identical input, then reports the max absolute error and the timing/speedup. No file I/O sits
// inside either timed region (CPU: chrono, GPU: CUDA events).
#include "heat2d.hpp"
#include "heat2d_cuda.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>

int main() {
    HeatParams p;
    p.width = 256;
    p.height = 256;
    p.lambda = 0.24;
    p.num_steps = 2000;
    p.snapshot_every = 100;
    p.out_dir = "snapshots";

    const std::size_t N = static_cast<std::size_t>(p.width) * p.height;
    const Grid initial = make_grid(p);

    //CPU reference (pure compute, no I/O).
    Grid cpu_cur = initial;
    Grid cpu_nxt(N, 0.0);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 1; s <= p.num_steps; ++s) {
        step_once(cpu_cur, cpu_nxt, p);
        std::swap(cpu_cur, cpu_nxt);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_s = std::chrono::duration<double>(t1 - t0).count();

    // GPU solver (pure compute timed via CUDA events)
    Grid gpu_final;
    float gpu_ms = 0.0f;
    solve_cuda(initial, gpu_final, p, &gpu_ms);
    const double gpu_s = gpu_ms / 1000.0;

    // I also added in correctness: max absolute error vs the CPU reference
    double max_err = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        max_err = std::max(max_err, std::abs(cpu_cur[k] - gpu_final[k]));
    }

    const double mcells =
        static_cast<double>(p.width) * p.height * p.num_steps / 1.0e6;

    std::cout << "Grid " << p.width << "x" << p.height
              << ", steps " << p.num_steps << ", lambda " << p.lambda << "\n";
    std::cout << "Max abs error (GPU vs CPU): " << max_err << "\n";
    std::cout << "CPU time: " << cpu_s << " s  (" << mcells / cpu_s
              << " Mcell-updates/s)\n";
    std::cout << "GPU time: " << gpu_s << " s  (" << mcells / gpu_s
              << " Mcell-updates/s)\n";
    std::cout << "Speedup (CPU/GPU): " << cpu_s / gpu_s << "x\n";
    return 0;
}
