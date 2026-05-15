// Necessary Standard Imports.
#pragma once
#include <string>
#include <vector>

// Numerical scheme (explicit, forward-Euler in time, central in space):
// Reminder while coding this hpp file:-
//   U[n+1][i,j] = (1 - 4 lambda) * U[n][i,j] + lambda * (U[n][i+1,j] + U[n][i-1,j] + U[n][i,j+1] + U[n][i,j-1])
//   where lambda = Gamma / Delta²  (time-step / space-step²).
//   Stability requires 0 < lambda < 0.5.

struct HeatParams {
    int    width;           // grid points along x (columns)
    int    height;          // grid points along y (rows)
    double lambda;          // lambda = dt/dx²; must satisfy 0 < lambda < 0.5
    int    num_steps;       // total explicit time steps to advance - an imp. param.
    int    snapshot_every;  // Basicalluy I dump a .dat file every this many steps
    std::string out_dir;    // Thus is an output directory for snapshot files, this is going to be gitginored.
};

// Flat, row-major storage for the 2D grid: U[i][j] lives at index i*width + j.
using Grid = std::vector<double>;

// Inline so the compiler can hoist it into the inner loop at zero cost.
inline int idx(int i, int j, int width) noexcept { return i * width + j; }

//Public API
// I allocate a zeroed grid and seed a single hot point at the centre.
Grid make_grid(const HeatParams& p);
// Advance one explicit time step from `cur` into `nxt`. Pls note: `cur` and `nxt` must not alias.
void step_once(const Grid& cur, Grid& nxt, const HeatParams& p);
// Finally for now a method to write `g` to "<out_dir>/snap_NNNNNN.dat" in gnuplot column format.
void write_snapshot(const Grid& g, const HeatParams& p, int step_num);
