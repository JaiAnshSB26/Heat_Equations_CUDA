// Bonus driver: implicit (backward-Euler) solver via Jacobi iteration.
// Validates the GPU implicit solver against a CPU implementation of the SAME
// scheme (so they must agree to round-off), reports timing, and demonstrates
// unconditional stability at a lambda well beyond the explicit limit of 1/4.
#include "heat2d.hpp"
#include "heat2d_implicit.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>

int main() {
    HeatParams p;
    p.width = 256;
    p.height = 256;
    p.lambda = 0.24;
    p.num_steps = 2000;
    p.snapshot_every = 100;
    p.out_dir = "snapshots";

    const int iters = 50;  // Jacobi sweeps per implicit step (converges at lam=0.24)
    const std::size_t N = static_cast<std::size_t>(p.width) * p.height;
    const Grid initial = make_grid(p);

    // ---- CPU reference (same scheme + iteration count) ----
    Grid cpu_out;
    const auto t0 = std::chrono::high_resolution_clock::now();
    solve_implicit_cpu(initial, cpu_out, p, iters);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_s = std::chrono::duration<double>(t1 - t0).count();

    // ---- GPU implicit (pure compute timed via CUDA events) ----
    Grid gpu_out;
    float gpu_ms = 0.0f;
    solve_implicit_cuda(initial, gpu_out, p, iters, &gpu_ms);
    const double gpu_s = gpu_ms / 1000.0;

    // ---- Correctness: GPU vs CPU implicit ----
    double max_err = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        max_err = std::max(max_err, std::abs(cpu_out[k] - gpu_out[k]));
    }

    std::cout << "=== Implicit scheme (backward-Euler, Jacobi) ===\n";
    std::cout << "Grid " << p.width << "x" << p.height << ", steps " << p.num_steps
              << ", lambda " << p.lambda << ", Jacobi iters/step " << iters << "\n";
    std::cout << "Max abs error (GPU vs CPU implicit): " << max_err << "\n";
    std::cout << "CPU time: " << cpu_s << " s\n";
    std::cout << "GPU time (events): " << gpu_s << " s\n";
    std::cout << "Speedup (CPU/GPU): " << cpu_s / gpu_s << "x\n";

    // ---- Unconditional-stability demonstration ----
    // lambda = 5.0 is 20x above the explicit stability limit (1/4); the explicit
    // scheme would diverge, but the implicit field must stay bounded in [0,1].
    HeatParams ps = p;
    ps.lambda = 5.0;
    ps.num_steps = 200;
    Grid stab_out;
    solve_implicit_cuda(initial, stab_out, ps, iters, nullptr);

    double mn = stab_out[0];
    double mx = stab_out[0];
    for (double v : stab_out) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    std::cout << "\nStability demo at lambda=" << ps.lambda << " ("
              << ps.num_steps << " steps): min=" << mn << ", max=" << mx
              << "  (bounded => stable; the explicit scheme would diverge here)\n";
    return 0;
}
