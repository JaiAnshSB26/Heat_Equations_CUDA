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
#include <vector>
#include <iomanip>

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

    std::cout << "\n=========================================================================================\n";
    std::cout << "                      EXTENDED SCALING PERFORMANCE BENCHMARK                     \n";
    std::cout << "=========================================================================================\n";
    std::cout << "| Grid Size | Steps | CPU Time (s) | CPU Mcells/s | GPU Time (s) | GPU Mcells/s | Speedup |\n";
    std::cout << "|-----------|-------|--------------|--------------|--------------|--------------|---------|\n";

    struct BenchCase {
        int size;
        int steps;
    };

    std::vector<BenchCase> cases = {
        {128,  5000},
        {256,  2000}, // Re-run to verify consistency
        {512,  1000},
        {1024,  500},
        {2048,  200},
        {4096,   50}
    };

    for (const auto& c : cases) {
        HeatParams bp;
        bp.width = c.size;
        bp.height = c.size;
        bp.lambda = p.lambda;
        bp.num_steps = c.steps;
        bp.snapshot_every = c.steps + 1; // Turn off file writing for benchmarks
        bp.out_dir = "snapshots";

        const std::size_t bN = static_cast<std::size_t>(bp.width) * bp.height;
        const Grid b_initial = make_grid(bp);

        // --- CPU ---
        Grid b_cpu_cur = b_initial;
        Grid b_cpu_nxt(bN, 0.0);
        auto bt0 = std::chrono::high_resolution_clock::now();
        for (int s = 1; s <= bp.num_steps; ++s) {
            step_once(b_cpu_cur, b_cpu_nxt, bp);
            std::swap(b_cpu_cur, b_cpu_nxt);
        }
        auto bt1 = std::chrono::high_resolution_clock::now();
        double b_cpu_s = std::chrono::duration<double>(bt1 - bt0).count();

        // --- GPU ---
        Grid b_gpu_final;
        float b_gpu_ms = 0.0f;
        solve_cuda(b_initial, b_gpu_final, bp, &b_gpu_ms);
        double b_gpu_s = b_gpu_ms / 1000.0;

        // --- Metrics ---
        double total_updates = static_cast<double>(bN) * bp.num_steps;
        double cpu_mcells = (total_updates / b_cpu_s) / 1e6;
        double gpu_mcells = (total_updates / b_gpu_s) / 1e6;
        double speedup = b_cpu_s / b_gpu_s;

        std::cout << "| " << c.size << "x" << c.size 
                  << " | " << std::setw(5) << c.steps
                  << " | " << std::fixed << std::setprecision(4) << std::setw(12) << b_cpu_s
                  << " | " << std::setprecision(1) << std::setw(12) << cpu_mcells
                  << " | " << std::setprecision(4) << std::setw(12) << b_gpu_s
                  << " | " << std::setprecision(1) << std::setw(12) << gpu_mcells
                  << " | " << std::setprecision(1) << std::setw(6) << speedup << "x"
                  << " |\n";
    }
    std::cout << "=========================================================================================\n";
    std::cout << "\n=========================================================================================\n";
    std::cout << "                          END-TO-END TIME (INCLUDING MEMORY COPIES)                       \n";
    std::cout << "=========================================================================================\n";
    
    // We reuse the parameters 'p' and 'initial' grid from your original 256x256 setup
    auto t_gpu_total_start = std::chrono::high_resolution_clock::now();

    Grid e2e_gpu_final;
    // solve_cuda internally handles cudaMalloc, cudaMemcpy H2D, the loop, and cudaMemcpy D2H
    solve_cuda(initial, e2e_gpu_final, p, nullptr); 

    auto t_gpu_total_end = std::chrono::high_resolution_clock::now();
    double gpu_total_seconds = std::chrono::duration<double>(t_gpu_total_end - t_gpu_total_start).count();

    // The benchmark table above left the stream sticky in std::fixed with
    // setprecision(1); reset so the end-to-end times print at full precision.
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "Original 256x256 Case (" << p.num_steps << " steps):\n";
    std::cout << "Sequential CPU Total Time                 : " << cpu_s << " s\n";
    // This is the raw loop time measured inside solve_cuda via CUDA events
    std::cout << "Pure GPU Compute Time (Events)            : " << gpu_s << " s\n"; 
    // This is the clock time measured by the CPU host around the whole function call
    std::cout << "Real-World GPU Time (Host Clock w/ Copies): " << gpu_total_seconds << " s\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";
    std::cout << "Pure Compute Speedup (Isolated)           : " << cpu_s / gpu_s << "x\n";
    std::cout << "Effective Real-World Speedup (End-to-End) : " << cpu_s / gpu_total_seconds << "x\n";
    std::cout << "=========================================================================================\n";
    
    return 0;
}
