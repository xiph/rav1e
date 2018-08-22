#!/bin/bash
set -e
rm -f rdo.dat
cargo build --release
ls ~/sets/objective-1-fast/*.y4m | parallel target/release/rav1e --threads 1 --quantizer {2} -o /dev/null --train-rdo {1} :::: - ::: 16 48 80 112 144 176 208 240
gnuplot rdo.plt -p
