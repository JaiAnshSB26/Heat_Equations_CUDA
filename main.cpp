#include "heat2d.hpp"
#include <iostream>

int main() {
    HeatParams p{
        256,
        256,
        0.1,
        1000,
        50,
        "snapshots"
    };

    Grid g = make_grid(p);

    std::cout << "Center value initialized.\n";

    return 0;
}