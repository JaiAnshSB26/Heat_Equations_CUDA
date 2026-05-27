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

void verify_solution(const Grid& g, const HeatParams& p, int step_num) {
    const int W = p.width;
    const int H = p.height;
    const int cx = W / 2;
    const int cy = H / 2;

    double max_val = 0.0;
    double min_val = 1.0;
    
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double val = g[idx(i, j, W)];
            if (val > max_val) max_val = val;
            if (val < min_val) min_val = val;
        }
    }

    if (min_val < 0.0 || max_val > 1.0) {
        std::printf("[CRITICAL ERROR] Step %d: Numerical instability detected! Min: %.4f, Max: %.4f\n", 
                    step_num, min_val, max_val);
        return;
    }

    int test_dist = 20; 
    double north = g[idx(cy - test_dist, cx, W)];
    double south = g[idx(cy + test_dist, cx, W)];
    double west  = g[idx(cy, cx - test_dist, W)];
    double east  = g[idx(cy, cx + test_dist, W)];

    double epsilon = 1e-7; // we add this in order to allow some potential minor differences
    if (std::abs(north - south) > epsilon || std::abs(east - west) > epsilon || std::abs(north - east) > epsilon) {
        std::printf("[ERROR] Step %d: Asymmetric diffusion detected! N:%.5f, S:%.5f, E:%.5f, W:%.5f\n",
                    step_num, north, south, east, west);
    } else {
        if (step_num % p.snapshot_every == 0 || step_num == p.num_steps) {
            std::printf("[VALIDATION] Step %4d | Max Temp (Center Decay): %.5f | Symmetry: PERFECT\n", 
                        step_num, max_val);
        }
    }
}