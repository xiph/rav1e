#!/bin/bash

# Stop on first error
set -e

# Move to the correct directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}/.."

echo dev profile: full build time
cargo clean
time cargo build -q

echo;echo;
echo dev profile: touch src/lib.rs
touch src/lib.rs
time cargo build -q

echo;echo;
echo release profile: full build time
cargo clean
time cargo build -q --release

echo;echo;
echo release profile: touch src/lib.rs
touch src/lib.rs
time cargo build -q --release
