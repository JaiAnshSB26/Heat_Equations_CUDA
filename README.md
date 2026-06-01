# Heat Equations in CUDA - Concurrent and Distributed Computing (CSE 305) Final Project.

## Authors - Jai Ansh Bindra (BX26) and Nicolas Sleiman (BX26)

## Instructors - M. Eric Goubault and M. Gleb Pogudin.

Important commands for the first part implementation of the project upto now:-

Part 1 currently implements a sequential C++17 solver for the 2D heat equation using an explicit finite-difference scheme.

### Build:-

```bash
make
```

### Run:-

```bash
make run
```

### Visualize:-

```bash
make plot
```

Please note: Outputs are written to `snapshots/` (not present in the repository since generated files are gitignored due to the reasons of file size and them just being generated outputs!).

The solver evolves the heat equation on a 2D grid using double buffering and exports snapshots compatible with gnuplot visualization.

## Part 3 - CUDA parallelization

The same explicit scheme is parallelized on the GPU: one CUDA thread computes one interior grid cell using flat row-major device arrays and on-device double buffering. Boundary cells are never written, so they stay fixed at 0 (matching the CPU path). Requires the NVIDIA CUDA Toolkit (`nvcc`).

### Build (CUDA):

```bash
make cuda
```

### Run (CUDA):

```bash
make run_cuda
```

`run_cuda` runs both the CPU reference and the GPU solver on identical input and prints the **max absolute error** (GPU vs CPU), per-path timings, and the speedup.
GPU compute time is measured with CUDA events and excludes host/device copies and file I/O. The shared gnuplot visualization (`make plot`) applies to both paths since the GPU output matches the CPU reference to machine precision.

### First Validation and Performance Result for the third part of the project.

Tested on:

- GPU: NVIDIA RTX 4000 Ada Generation
- CUDA Toolkit: 12.9
- Grid size: 256 × 256
- Time steps: 2000
- λ = 0.24

Representative run:

```text
Max abs error (GPU vs CPU): 1.11022e-16
CPU time: 0.0991978 s
GPU time: 0.00834592 s
Speedup (CPU/GPU): 11.8858x
```

The GPU implementation matches the CPU reference to machine precision while achieving an approximately 11× speedup on the test configuration.