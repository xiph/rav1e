FROM mcr.microsoft.com/devcontainers/rust:1-1-bullseye
RUN sed -i 's/bullseye/testing/g' /etc/apt/sources.list \
 && dpkg --add-architecture arm64 \
 && apt-get update && export DEBIAN_FRONTEND=noninteractive \
 && apt-get install -y --no-install-recommends \
 libaom-dev libdav1d-dev libaom-dev:arm64 libdav1d-dev:arm64 \
 libclang-dev libgit2-dev libcurl4-openssl-dev libfuzzer-14-dev \
 gcc-aarch64-linux-gnu libc6-dev-arm64-cross nasm dav1d qemu-user \
 && rm -rf /var/lib/apt/lists/*
RUN rustup target add aarch64-unknown-linux-gnu
RUN export CARGO_PROFILE_RELEASE_STRIP=true \
 && cargo install -q cargo-c \
 && cargo install -q cargo-criterion \
 && cargo install -q cargo-fuzz \
 && rm -rf /usr/local/cargo/registry
RUN SDE=sde-external-9.14.0-2022-10-25-lin \
 && curl -sSf https://downloadmirror.intel.com/751535/$SDE.tar.xz | tar Jx -C /opt \
 && ln -sv /opt/$SDE/sde64 /usr/local/bin/sde64
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER=qemu-aarch64
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS=-Clinker=aarch64-linux-gnu-gcc
ENV PKG_CONFIG_ALLOW_CROSS_aarch64_unknown_linux_gnu=1
ENV PKG_CONFIG_PATH_aarch64_unknown_linux_gnu=/usr/lib/aarch64-linux-gnu/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu=/
ENV CUSTOM_LIBFUZZER_PATH=/usr/lib/llvm-14/lib/libFuzzer.a
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -icl --"
LABEL dev.containers.source=https://github.com/xiph/rav1e
LABEL dev.containers.id=rav1e