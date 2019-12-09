#!/usr/bin/env python3
import struct
import sys
from os.path import splitext
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# Renders block importances output by ContextInner::compute_lookahead_data().
# Usage:
#        cargo run --features=dump_lookahead_data <input.y4m> -o /dev/null
#        draw-importances.py <i-hres.png> <i-imps.bin>
#        will output i-imps.png.


with open(sys.argv[2], 'rb') as f:
    contents = f.read()

rows, cols = struct.unpack('qq', contents[:16])
imps = np.frombuffer(contents[16:], dtype=np.float32).reshape((rows, cols))

# Use a fixed scale where anything >= 10 cannot be distinguished
# to allow visually comparing multiple pictures
max_imp = 10 ## Replace by `np.max(imps)` for relative scaling

frame_size_multiplier = 4

frame = Image.open(sys.argv[1])
frame = frame.resize((frame.width * frame_size_multiplier, frame.height * frame_size_multiplier))
frame = frame.convert(mode='RGB')

mv_original_block_size = 8 // 2 # The importances are in 8×8 blocks, but we use half-resolution images.
mv_block_size = mv_original_block_size * frame_size_multiplier

draw = ImageDraw.Draw(frame, mode='RGBA')

# Draw the grid.
for i in range(0, frame.width, mv_block_size):
    draw.line(((i, 0), (i, frame.height)), fill=(0, 0, 0, 255))
for i in range(0, frame.height, mv_block_size):
    draw.line(((0, i), (frame.width, i)), fill=(0, 0, 0, 255))

# Draw the importances.
if max_imp > 0:
    for y in range(rows):
        for x in range(cols):
            imp = imps[y, x]
            top_left = (x * mv_block_size, y * mv_block_size)
            bottom_right = (top_left[0] + mv_block_size, top_left[1] + mv_block_size)
            draw.rectangle((top_left, bottom_right), fill=(int(imp / max_imp * 255), 0, 0, 128))

fig = plt.figure(figsize=(frame.width, frame.height), dpi=1)
fig.figimage(frame, cmap='gray')
plt.savefig(splitext(sys.argv[2])[0] + '.png', bbox_inches='tight')
