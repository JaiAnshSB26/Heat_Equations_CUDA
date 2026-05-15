# plot.gp  –  visualise heat equation snapshots produced by heat2d

STEPS  = 2000
EVERY  = 100

W = 255
H = 255

set palette defined ( \
    0.00 "#000000", \
    0.10 "#00008B", \
    0.40 "#8B0000", \
    0.70 "#FF4500", \
    1.00 "#FFFF00"  \
)

set xrange [0:W]
set yrange [0:H]

unset key

set xlabel "x"
set ylabel "y"

set size ratio -1

CMAX = 0.10
set cbrange [0:CMAX]
set cblabel "Temperature"

# PNG FRAMES

set terminal pngcairo size 700,640 enhanced font "Arial,12"

do for [s = 0:STEPS:EVERY] {

    fname = sprintf("snapshots/snap_%06d.dat", s)

    set output sprintf("snapshots/snap_%06d.png", s)

    set title sprintf("Heat diffusion - step %d", s)

    plot fname using 1:2:3 with image pixels
}

# ANIMATED GIF

set terminal gif animate delay 8 size 700,640 optimize

set output "snapshots/heat_anim.gif"

do for [s = 0:STEPS:EVERY] {

    fname = sprintf("snapshots/snap_%06d.dat", s)

    set title sprintf("Heat diffusion - step %d", s)

    plot fname using 1:2:3 with image pixels
}

set output

print "Generated PNG frames and animated GIF."