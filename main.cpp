#include "heat2d.hpp"

#include <iostream>
#include <utility>

int main() {
    HeatParams p;
    p.width = 256;
    p.height = 256;
    p.lambda = 0.24;
    p.num_steps = 200;
    p.snapshot_every = 50;
    p.out_dir = "snapshots";

    Grid cur = make_grid(p);
    Grid nxt(p.width * p.height, 0.0);

    write_snapshot(cur, p, 0);

    for (int s = 1; s <= p.num_steps; ++s) {
        step_once(cur, nxt, p);
        std::swap(cur, nxt);

        if (s % p.snapshot_every == 0) {
            write_snapshot(cur, p, s);
            std::cout << "Wrote snapshot at step " << s << "\n";
        }
    }

    std::cout << "Finished sequential heat simulation.\n";
    return 0;
}