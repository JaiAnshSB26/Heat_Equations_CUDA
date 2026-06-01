#include "heat2d.hpp"

#include <iostream>
#include <utility>
#include <chrono>

int main() {
    HeatParams p;
    p.width = 256;
    p.height = 256;
    p.lambda = 0.24;
    p.num_steps = 2000; //200.
    p.snapshot_every = 100; //50.
    p.out_dir = "snapshots";

    Grid cur = make_grid(p);
    Grid nxt(p.width * p.height, 0.0);

    write_snapshot(cur, p, 0);

    const auto start = std::chrono::high_resolution_clock::now(); //important addition for the runtime measurement.

    for (int s = 1; s <= p.num_steps; ++s) {
        step_once(cur, nxt, p);
        std::swap(cur, nxt);

        verify_solution(cur, p, s);

        if (s % p.snapshot_every == 0) {
            write_snapshot(cur, p, s);
            std::cout << "Wrote snapshot at step " << s << "\n";
        }
    }

    const auto end = std::chrono::high_resolution_clock::now(); //for runtime measurement again.
    const std::chrono::duration<double> elapsed = end - start;

    // std::cout << "Total runtime: " << elapsed.count() << " seconds\n";
    // std::cout << "Finished sequential heat simulation.\n";

    const double mcells =
        static_cast<double>(p.width) * p.height * p.num_steps / 1.0e6;

    std::cout << "Total runtime: " << elapsed.count() << " seconds\n";
    std::cout << "Throughput: " << mcells / elapsed.count()
            << " Mcell-updates/s\n";
    std::cout << "Finished sequential heat simulation.\n";
    return 0;
}