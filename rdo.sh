#!/bin/bash
set -e
rm -f rdo.dat
cargo build --release
ls ~/sets/subset1/*.y4m | parallel target/release/rav1e -s 0 --quantizer 80 -o /dev/null --train-rdo
gnuplot rdo.plt -p
