#include "heat2d.hpp"

#include <iostream>
#include <utility>

int main() {
    HeatParams p;
    p.width = 256;
    p.height = 256;
    p.lambda = 0.24;
    p.num_steps = 10;
    p.snapshot_every = 100;
    p.out_dir = "snapshots";

    Grid cur = make_grid(p);
    Grid nxt(p.width * p.height, 0.0);

    for (int s = 0; s < p.num_steps; ++s) {
        step_once(cur, nxt, p);
        std::swap(cur, nxt);
    }

    std::cout << "Ran " << p.num_steps << " sequential heat steps.\n";
    return 0;
}