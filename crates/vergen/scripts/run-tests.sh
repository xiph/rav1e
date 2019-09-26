#!/bin/bash
set -ev

if [ "${TRAVIS_RUST_VERSION}" = "stable" ]; then
    cargo build
    cargo test
elif [ "${TRAVIS_RUST_VERSION}" = "beta" ]; then
    cargo build
    cargo test
else
    cargo build
    cargo test
fi
