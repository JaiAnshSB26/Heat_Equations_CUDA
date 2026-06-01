// CUDA acceleration of the explicit 2D heat-equation scheme.
// Firstly, the naive global-memory kernel, one GPU thread per interior cell.
#pragma once
#include "heat2d.hpp"
// Advance one explicit time step on the GPU (naive baseline).
// We allocate device buffers, copy `cur` in, launch the kernel, copy the result back, then free. `cur` and `nxt` must not alias.
// Important: Boundary cells are never written, so they stay fixed at 0.(would be useful soon.)
void step_once_cuda(const Grid& cur, Grid& nxt, const HeatParams& p);