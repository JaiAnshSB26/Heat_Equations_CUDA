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