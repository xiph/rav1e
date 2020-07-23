import struct
import sys

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

# Renders motion vectors output by ContextInner::compute_lookahead_data().
# Requires *-hres.png and *-mvs.bin files.
# Usage: draw-mvs.py <number>
#        will read i-hres.png and i-mvs.bin for i in [0; number) and
#        output i-mvs.png.


def draw_mvs(prefix):
    with open(prefix + "-mvs.bin", "rb") as f:
        contents = f.read()

    rows, cols = struct.unpack("qq", contents[:16])
    mvs = np.frombuffer(contents[16:], dtype=np.short).reshape((rows, cols, 2))

    frame_size_multiplier = 4

    frame = Image.open(prefix + "-hres.png")
    frame = frame.resize(
        (
            frame.width * frame_size_multiplier,
            frame.height * frame_size_multiplier,
        )
    )

    mv_original_block_size = (
        4 // 2
    )  # The MVs are in 4×4 blocks, but we use half-resolution images.
    mv_subsampling = (
        4  # The MVs currently computed are the same in 4×4 blocks.
    )
    mv_units_per_pixel = 8  # MVs are in 8ths of a pixel.
    mvs = mvs[::mv_subsampling, ::mv_subsampling]
    rows = rows // mv_subsampling
    cols = cols // mv_subsampling
    mv_block_size = (
        mv_original_block_size * frame_size_multiplier * mv_subsampling
    )

    draw = ImageDraw.Draw(frame)

    # Draw the grid.
    for i in range(0, frame.width, mv_block_size):
        draw.line(((i, 0), (i, frame.height)))
    for i in range(0, frame.height, mv_block_size):
        draw.line(((0, i), (frame.width, i)))

    # Draw the motion vectors.
    for y in range(rows):
        for x in range(cols):
            mv = mvs[y, x]
            start = (
                x * mv_block_size + mv_block_size // 2,
                y * mv_block_size + mv_block_size // 2,
            )
            end = (
                mv[1] * frame_size_multiplier // mv_units_per_pixel + start[0],
                mv[0] * frame_size_multiplier // mv_units_per_pixel + start[1],
            )
            draw.line((start, end), fill=255)

    fig = plt.figure(figsize=(frame.width, frame.height), dpi=1)
    fig.figimage(frame, cmap="gray")
    plt.savefig(prefix + "-mvs.png", bbox_inches="tight")
    plt.close(fig)


for prefix in range(1, int(sys.argv[1])):
    print(prefix)
    draw_mvs(str(prefix))
