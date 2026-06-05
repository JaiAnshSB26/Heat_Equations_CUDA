// Optional / bonus: implicit (backward-Euler) heat solver.
// The implicit scheme is unconditionally stable; each time step solves the
// sparse linear system (1+4*lam)U^{n+1} - lam*(neighbours) = U^n, which we do
// with Jacobi iteration. A Jacobi sweep is the same five-point stencil as the
// explicit kernel, so it is embarrassingly parallel on the GPU.
#pragma once
#include "heat2d.hpp"

// CPU reference: backward-Euler advanced by `iters` Jacobi sweeps per time step.
// Boundaries stay fixed at 0. `iters` should be large enough to converge the
// inner solve (the diagonally dominant system makes Jacobi convergent).
void solve_implicit_cpu(const Grid& initial, Grid& out,
                        const HeatParams& p, int iters);

// GPU version of the same scheme/iteration count, run entirely on the device
// with double-buffered Jacobi sweeps. If `gpu_ms` is non-null it receives the
// pure compute time in milliseconds (CUDA events; copies excluded).
void solve_implicit_cuda(const Grid& initial, Grid& out,
                         const HeatParams& p, int iters, float* gpu_ms = nullptr);
