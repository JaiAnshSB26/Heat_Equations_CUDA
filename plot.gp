set terminal pngcairo size 900,700
set output "snapshots/heatmap_000200.png"

set title "2D Heat Equation - step 200"
set xlabel "x"
set ylabel "y"

set size ratio -1
set pm3d map
unset key

set palette defined ( \
    0.00 "#000000", \
    0.10 "#00008B", \
    0.40 "#8B0000", \
    0.70 "#FF4500", \
    1.00 "#FFFF00"  \
)

set cbrange [0:0.10]
set cblabel "Temperature"

plot "snapshots/snap_000200.dat" using 1:2:3 with image pixels