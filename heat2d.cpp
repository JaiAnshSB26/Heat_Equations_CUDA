#include "heat2d.hpp"

#include <cassert>

Grid make_grid(const HeatParams& p) {
    Grid g(p.width * p.height, 0.0);

    // Single hot point at the centre.
    const int center_row = p.height / 2;
    const int center_col = p.width / 2;

    g[idx(center_row, center_col, p.width)] = 1.0;

    return g;
}

void step_once(const Grid& cur, Grid& nxt, const HeatParams& p) {
    assert(cur.size() == nxt.size());

    const double lam = p.lambda;
    const double self = 1.0 - 4.0 * lam;

    const int W = p.width;
    const int H = p.height;

    for (int i = 1; i < H - 1; ++i) {
        for (int j = 1; j < W - 1; ++j) {
            nxt[idx(i, j, W)] =
                self * cur[idx(i, j, W)]
                + lam * cur[idx(i - 1, j, W)]
                + lam * cur[idx(i + 1, j, W)]
                + lam * cur[idx(i, j - 1, W)]
                + lam * cur[idx(i, j + 1, W)];
        }
    }
}

void write_snapshot(const Grid&, const HeatParams&, int) {
    // I would implement this later.
}