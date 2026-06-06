# Heat Equations in CUDA - Concurrent and Distributed Computing (CSE305) Final Project

## Authors

- Jai Ansh Bindra (BX26)
- Nicolas Sleiman (BX26)

## Instructors

- M. Eric Goubault
- M. Gleb Pogudin

---

## Project Overview

This project studies numerical solutions of the 2D heat equation and their acceleration using NVIDIA CUDA.

Starting from a sequential C++ implementation of an explicit finite-difference scheme, the project progressively introduces validation procedures, GPU parallelization, and performance benchmarking against the CPU reference implementation.

---

## The Heat Equation

We approximate solutions of the 2D heat equation

```text
∂U/∂t = ∂²U/∂x² + ∂²U/∂y²
```

using the explicit forward-Euler finite-difference scheme:

```text
U[n+1][i,j] = (1 - 4λ) · U[n][i,j]
            + λ · (U[n][i+1,j] + U[n][i-1,j]
            + U[n][i,j+1] + U[n][i,j-1])
```

where

```text
λ = Γ / Δ²
```

The explicit scheme is stable for:

```text
0 < λ ≤ 1/4 (and therefore λ < 0.5 as required in the project specification)
```

In our experiments we use:

```text
λ = 0.24
```

---

## Sequential Implementation

The sequential solver is implemented in C++17.

Main design choices:

- Flat row-major grid representation using `std::vector<double>`
- Explicit finite-difference stencil
- Double buffering (current grid / next grid)
- Fixed zero-valued boundary cells (Dirichlet boundary conditions)
- Snapshot export for visualization
- Runtime and throughput measurements

The initial condition consists of a circular hot region centered in the grid.

### Build

```bash
make
```

### Run

```bash
make run
```

### Visualize

```bash
make plot
```

Generated outputs are written to the `snapshots/` directory.

---

## Validation

The numerical solution is validated through:

- Temperature bound checks
- Symmetry preservation checks
- Monitoring of maximum temperature decay
- CPU/GPU result comparison through maximum absolute error
- Stability verification for admissible λ values

Example validation output:

```text
[VALIDATION] Step 2000 | Max Temp (Center Decay): 0.07049 | Symmetry: PERFECT
```

---

## CUDA Parallelization

The explicit stencil is parallelized using NVIDIA CUDA.

Main design choices:

- One CUDA thread computes one interior grid cell
- Flat row-major device arrays
- On-device double buffering
- Fixed zero-valued boundary cells
- CUDA event timing
- CPU implementation retained as the reference solution

### Build (CUDA)

```bash
make cuda
```

### Run (CUDA)

```bash
make run_cuda
```

The CUDA executable runs both CPU and GPU versions on identical input and reports:

- Maximum absolute error
- CPU runtime
- GPU runtime
- Throughput
- Speedup

---

## Validation and Performance Results

Test platform:

- GPU: NVIDIA RTX 4000 Ada Generation
- CUDA Toolkit: 12.9

Configuration:

- Grid size: 256 x 256
- Time steps: 2000
- λ = 0.24

Representative result:

```text
Max abs error (GPU vs CPU): 1.11022e-16

CPU time: 0.0991978 s
GPU time: 0.00834592 s

Speedup (CPU/GPU): 11.8858x
```

The GPU implementation matches the CPU reference to machine precision while achieving approximately an 11x-12x speedup on this configuration.

---

## Benchmarking

The benchmark suite evaluates performance across multiple grid sizes and simulation lengths and measures:

- CPU runtime
- GPU runtime
- Throughput
- Speedup
- Numerical error

for multiple grid sizes and simulation lengths.

These measurements are used to study scaling behaviour and compare sequential and GPU implementations.

---

## Optional Bonus: Implicit Solver

An optional implicit (backward-Euler) version of the heat equation was implemented using Jacobi iteration.

Unlike the explicit scheme, the implicit scheme is unconditionally stable and allows significantly larger values of λ. Each implicit time step solves the sparse linear system using repeated Jacobi sweeps, which are parallelized on the GPU with the same one-thread-per-cell stencil structure as the explicit CUDA solver.

### Build

```bash
make implicit_cuda
```

### Run

```bash
make run_implicit_cuda
```

Representative implicit result:

```text
Max abs error (GPU vs CPU implicit): 1.80411e-16

CPU time: 4.6917 s
GPU time (events): 0.644205 s

Speedup (CPU/GPU): 7.28293x

Stability demo at λ = 5: min = 0, max = 0.0378437
```

---

## Notes

- Generated snapshots are excluded from the repository through `.gitignore`.
- CPU and GPU implementations are validated against one another before benchmarking.
- The accompanying report discusses the numerical scheme, CUDA design choices, validation methodology, and performance analysis in detail.