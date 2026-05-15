#include "heat2d.hpp"

Grid make_grid(const HeatParams& p)
{
    Grid g(p.width * p.height, 0.0);

    // Single hot point at the centre.
    const int center_row = p.height / 2;
    const int center_col = p.width / 2;

    g[idx(center_row, center_col, p.width)] = 100.0;

    return g;
}