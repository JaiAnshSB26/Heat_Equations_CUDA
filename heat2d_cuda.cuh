// CUDA acceleration of the explicit 2D heat-equation scheme.
// Firstly, the naive global-memory kernel, one GPU thread per interior cell.
#pragma once
#include "heat2d.hpp"
// Advance one explicit time step on the GPU (naive baseline).
// We allocate device buffers, copy `cur` in, launch the kernel, copy the result back, then free. `cur` and `nxt` must not alias.
// Important: Boundary cells are never written, so they stay fixed at 0.(would be useful soon.)
void step_once_cuda(const Grid& cur, Grid& nxt, const HeatParams& p);
// Now we need to run the full simulation (p.num_steps) on the GPU for which we allocate two device buffers once, copy `initial` in, ping-pong them with
// on-device double buffering, then copy the final grid into `out`.
// Again, boundaries stay fixed at 0 throughout. The CPU path remains the reference.
void solve_cuda(const Grid& initial, Grid& out, const HeatParams& p, float* gpu_ms = nullptr);
// Now if gpu_ms is non-null, it receives the pure kernel-loop time in milliseconds, (measured with CUDA events.)
