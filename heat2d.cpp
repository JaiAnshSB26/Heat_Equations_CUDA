#include "heat2d.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>

Grid make_grid(const HeatParams& p) {
    Grid g(p.width * p.height, 0.0);
    
    // Single hot point at the centre.
    const int cx = p.width / 2;
    const int cy = p.height / 2;

    const int radius = 12;

    for (int i = 0; i < p.height; ++i) {
        for (int j = 0; j < p.width; ++j) {

            int dx = j - cx;
            int dy = i - cy;

            if (dx * dx + dy * dy <= radius * radius) {
                g[idx(i, j, p.width)] = 1.0;
            }
        }
    }

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

void write_snapshot(const Grid& g, const HeatParams& p, int step_num) {
    std::filesystem::create_directories(p.out_dir);

    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/snap_%06d.dat",
                  p.out_dir.c_str(), step_num);

    FILE* f = std::fopen(fname, "w");
    if (!f) {
        std::perror(fname);
        return;
    }
    for (int i = 0; i < p.height; ++i) {
        for (int j = 0; j < p.width; ++j) {
            std::fprintf(f, "%d %d %.8f\n", j, i, g[idx(i, j, p.width)]);
        }
        std::fputc('\n', f);
    }
    std::fclose(f);
}