FROM gcr.io/oss-fuzz-base/base-builder-rust
RUN apt-get update && apt-get install --no-install-recommends -y \
        clang curl llvm-dev libclang-dev pkg-config \
	python3-pip python3-setuptools python3-wheel nasm && \
    pip3 install meson ninja
RUN curl -sLO http://get.videolan.org/dav1d/1.0.0/dav1d-1.0.0.tar.xz && \
    tar Jxf dav1d-1.0.0.tar.xz && mv dav1d-1.0.0 dav1d && rm dav1d-1.0.0.tar.xz
COPY . $SRC/rav1e
WORKDIR $SRC/rav1e
COPY ./.clusterfuzzlite/build.sh ./.clusterfuzzlite/*.options $SRC/
