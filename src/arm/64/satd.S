/* Copyright (c) 2022, The rav1e contributors. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "src/arm/asm.S"
#include "util.S"

.macro butterfly r0, r1, r2, r3
    add  \r0\().8h, \r2\().8h, \r3\().8h
    sub  \r1\().8h, \r2\().8h, \r3\().8h
.endm

.macro interleave r0, r1, r2, r3
    zip1 \r0\().8h, \r2\().8h, \r3\().8h
    zip2 \r1\().8h, \r2\().8h, \r3\().8h
.endm

.macro interleave_pairs r0, r1, r2, r3
    zip1 \r0\().4s, \r2\().4s, \r3\().4s
    zip2 \r1\().4s, \r2\().4s, \r3\().4s
.endm

.macro normalize_4
    add     w0, w0, 2
    lsr     w0, w0, 2
.endm

// x0: src: *const u8,
// x1: src_stride: isize,
// x2: dst: *const u8,
// x3: dst_stride: isize,
function satd4x4_neon, export=1
    #define src         x0
    #define src_stride  x1
    #define dst         x2
    #define dst_stride  x3

    ldr     s0, [src]
    ldr     s1, [dst]

    // subtract; cast to 16-bit
    usubl v0.8h, v0.8b, v1.8b

    ldr     s1, [src, src_stride]
    ldr     s2, [dst, dst_stride]

    usubl v1.8h, v1.8b, v2.8b

    // stride * 2
    lsl     x8, src_stride, 1
    lsl     x9, dst_stride, 1

    ldr     s2, [src, x8]
    ldr     s3, [dst, x9]

    usubl v2.8h, v2.8b, v3.8b

    // stride * 3
    add     x8, src_stride, src_stride, lsl 1
    add     x9, dst_stride, dst_stride, lsl 1

    ldr     s3, [src, x8]
    ldr     s4, [dst, x9]

    usubl v3.8h, v3.8b, v4.8b

    // pack rows 0-2, 1-3
    mov   v0.d[1], v2.d[0]
    mov   v1.d[1], v3.d[0]

    // Horizontal transform

    // v0    0 1 2 3   8  9 10 11
    // v1    4 5 6 7  12 13 14 15

    butterfly v2, v3, v0, v1

    // v2    [0+4][1+5][2+6][3+7] [8+12][9+13][10+14][11+15]
    // v3    [0-4][1-5][2-6][3-7] [8-12][9-13][10-14][11-15]

    interleave v0, v1, v2, v3

    // v0    [ 0+4][ 0-4][ 1+5][ 1-5] [2 + 6][2 - 6][3 + 7][3 - 7]
    // v1    [8+12][8-12][9+13][9-13] [10+14][10-14][11+15][11-15]

    butterfly v2, v3, v0, v1

    // v2    [0+4+8+12][0-4+8-12][1+5+9+13][1-5+9-13] [2+6+10+14][2-6+10-14][3+7+11+15][3-7+11-15]
    // v3    [0+4-8-12][0-4-8+12][1+5-9-13][1-5-9+13] [2+6-10-14][2-6-10+14][3+7-11-15][3-7-11+15]

    interleave_pairs v0, v1, v2, v3

    // Vertical transform

    butterfly v2, v3, v0, v1

    interleave v0, v1, v2, v3

    butterfly v2, v3, v0, v1

    // sum up transform
    abs   v2.8h, v2.8h
    abs   v3.8h, v3.8h

    add v0.8h, v2.8h, v3.8h

    addv    h0, v0.8h
    fmov    w0, s0
    normalize_4
    ret

    #undef src
    #undef src_stride
    #undef dst
    #undef dst_stride
endfunc

.macro DOUBLE_HADAMARD_4X4
    // Horizontal transform

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5

    interleave v0, v1, v2, v3
    interleave v4, v5, v6, v7

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5

    interleave_pairs v0, v1, v2, v3
    interleave_pairs v4, v5, v6, v7

    // Vertical transform

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5

    interleave v0, v1, v2, v3
    interleave v4, v5, v6, v7

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5
.endm

.macro SUM_DOUBLE_HADAMARD_4X4
    abs     v2.8h, v2.8h
    abs     v3.8h, v3.8h
    abs     v6.8h, v6.8h
    abs     v7.8h, v7.8h

    add     v0.8h, v2.8h, v3.8h
    add     v1.8h, v6.8h, v7.8h
    add     v0.8h, v0.8h, v1.8h

    addv    h0, v0.8h
    fmov    w0, s0
    normalize_4
.endm

function satd8x4_neon, export=1
    #define src         x0
    #define src_stride  x1
    #define dst         x2
    #define dst_stride  x3

    // load 8 pixel row
    ldr     d0, [src]
    ldr     d1, [dst]

    usubl v0.8h, v0.8b, v1.8b

    ldr     d1, [src, src_stride]
    ldr     d2, [dst, dst_stride]

    usubl v1.8h, v1.8b, v2.8b

    lsl     x8, src_stride, 1
    lsl     x9, dst_stride, 1

    ldr     d2, [src, x8]
    ldr     d3, [dst, x9]

    usubl v2.8h, v2.8b, v3.8b

    // stride * 3
    add     x8, src_stride, src_stride, lsl 1
    add     x9, dst_stride, dst_stride, lsl 1

    ldr     d3, [src, x8]
    ldr     d4, [dst, x9]

    usubl v3.8h, v3.8b, v4.8b

    // extract top 64 bits out of register
    // (4 x 16 bits = 64)

    ext v4.16b, v0.16b, v0.16b, 8
    ext v5.16b, v1.16b, v1.16b, 8

    // pack rows 0-2, 1-3 (set 1)
    mov   v0.d[1], v2.d[0]
    mov   v1.d[1], v3.d[0]

    // pack rows 0-2, 1-3 (set 2)
    mov   v4.d[1], v2.d[1]
    mov   v5.d[1], v3.d[1]

    // v2-3 temp registers for first 4x4 block//
    // 6-7 for second block

    DOUBLE_HADAMARD_4X4

    SUM_DOUBLE_HADAMARD_4X4
    ret

    #undef src
    #undef src_stride
    #undef dst
    #undef dst_stride
endfunc

.macro load_row n0, n1, src, dst, src_stride, dst_stride, should_add=1
    ldr     s\n0, [\src]
    ldr     s\n1, [\dst]

    usubl v\n0\().8h, v\n0\().8b, v\n1\().8b

.if \should_add != 0
    add     \src, \src, \src_stride
    add     \dst, \dst, \dst_stride
.endif
.endm

.macro load_row2 n0, n1, src, dst, src_stride, dst_stride
    ldr     s\n0, [\src, \src_stride]
    ldr     s\n1, [\dst, \dst_stride]

    usubl v\n0\().8h, v\n0\().8b, v\n1\().8b
.endm

function satd4x8_neon, export=1
    #define src         x0
    #define src_stride  x1
    #define dst         x2
    #define dst_stride  x3

    // 0 * stride
    load_row    0, 1, src, dst, src_stride, dst_stride, 0
    // 1 * stride
    load_row2   1, 2, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    // pattern repeats
    load_row    2, 3, src, dst, src_stride, dst_stride, 0
    load_row2   3, 4, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    4, 5, src, dst, src_stride, dst_stride, 0
    load_row2   5, 6, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    6, 7, src, dst, src_stride, dst_stride, 0
    load_row2   7, 8, src, dst, src_stride, dst_stride

    // pack rows
    mov   v0.d[1], v2.d[0]
    mov   v1.d[1], v3.d[0]

    mov   v4.d[1], v6.d[0]
    mov   v5.d[1], v7.d[0]

    DOUBLE_HADAMARD_4X4

    SUM_DOUBLE_HADAMARD_4X4
    ret

    #undef src
    #undef src_stride
    #undef dst
    #undef dst_stride
endfunc

function satd16x4_neon, export=1
    #define src         x0
    #define src_stride  x1
    #define dst         x2
    #define dst_stride  x3

    #define ROW1 v0
    #define ROW2 v1
    #define TMP1 v2
    #define TMP2 v3

    #define ROW3 v4
    #define ROW4 v5
    #define TMP3 v6
    #define TMP4 v7

    #define ROW5 v8
    #define ROW6 v9
    #define TMP5 v12
    #define TMP6 v13

    #define ROW7 v10
    #define ROW8 v11
    #define TMP7 v14
    #define TMP8 v15

    // load 16 pixel row
    ldr     q0, [src]
    ldr     q1, [dst]

    usubl2  v8.8h, v0.16b, v1.16b
    usubl   v0.8h, v0.8b, v1.8b

    ldr     q1, [src, src_stride]
    ldr     q2, [dst, dst_stride]

    usubl2  v9.8h, v1.16b, v2.16b
    usubl   v1.8h, v1.8b, v2.8b

    lsl     x8, src_stride, 1
    lsl     x9, dst_stride, 1

    ldr     q2, [src, x8]
    ldr     q3, [dst, x9]

    usubl2  v6.8h, v2.16b, v3.16b
    usubl   v2.8h, v2.8b, v3.8b

    // stride * 3
    add     x8, src_stride, src_stride, lsl 1
    add     x9, dst_stride, dst_stride, lsl 1

    ldr     q3, [src, x8]
    ldr     q4, [dst, x9]

    usubl2  v7.8h, v3.16b, v4.16b
    usubl   v3.8h, v3.8b, v4.8b

    // swap high/low 64 bits
    ext v4.16b, v0.16b, v0.16b, 8
    ext v5.16b, v1.16b, v1.16b, 8

    mov   v0.d[1], v2.d[0]
    mov   v1.d[1], v3.d[0]

    ext v10.16b, v8.16b, v8.16b, 8
    ext v11.16b, v9.16b, v9.16b, 8

    mov   v8.d[1], v6.d[0]
    mov   v9.d[1], v7.d[0]
    // 2-3 free

    mov   v4.d[1], v2.d[1]
    mov   v5.d[1], v3.d[1]
    // 6-7 free
    mov   v10.d[1], v6.d[1]
    mov   v11.d[1], v7.d[1]

    // 0,1       2,3
    // 4,5       6,7
    // 8,9       12,13
    // 10,11     14,15

    // quadruple 4x4 hadamard

    butterfly TMP1, TMP2, ROW1, ROW2
    butterfly TMP3, TMP4, ROW3, ROW4
    butterfly TMP5, TMP6, ROW5, ROW6
    butterfly TMP7, TMP8, ROW7, ROW8

    interleave ROW1, ROW2, TMP1, TMP2
    interleave ROW3, ROW4, TMP3, TMP4
    interleave ROW5, ROW6, TMP5, TMP6
    interleave ROW7, ROW8, TMP7, TMP8

    butterfly TMP1, TMP2, ROW1, ROW2
    butterfly TMP3, TMP4, ROW3, ROW4
    butterfly TMP5, TMP6, ROW5, ROW6
    butterfly TMP7, TMP8, ROW7, ROW8

    interleave_pairs ROW1, ROW2, TMP1, TMP2
    interleave_pairs ROW3, ROW4, TMP3, TMP4
    interleave_pairs ROW5, ROW6, TMP5, TMP6
    interleave_pairs ROW7, ROW8, TMP7, TMP8

    butterfly TMP1, TMP2, ROW1, ROW2
    butterfly TMP3, TMP4, ROW3, ROW4
    butterfly TMP5, TMP6, ROW5, ROW6
    butterfly TMP7, TMP8, ROW7, ROW8

    interleave ROW1, ROW2, TMP1, TMP2
    interleave ROW3, ROW4, TMP3, TMP4
    interleave ROW5, ROW6, TMP5, TMP6
    interleave ROW7, ROW8, TMP7, TMP8

    butterfly TMP1, TMP2, ROW1, ROW2
    butterfly TMP3, TMP4, ROW3, ROW4
    butterfly TMP5, TMP6, ROW5, ROW6
    butterfly TMP7, TMP8, ROW7, ROW8

    // absolute value of transform coefficients
    abs  TMP1.8h, TMP1.8h
    abs  TMP2.8h, TMP2.8h
    abs  TMP3.8h, TMP3.8h
    abs  TMP4.8h, TMP4.8h
    abs  TMP5.8h, TMP5.8h
    abs  TMP6.8h, TMP6.8h
    abs  TMP7.8h, TMP7.8h
    abs  TMP8.8h, TMP8.8h

    // stage 1 sum
    add TMP1.8h, TMP1.8h, TMP5.8h
    add TMP2.8h, TMP2.8h, TMP6.8h
    add TMP3.8h, TMP3.8h, TMP7.8h
    add TMP4.8h, TMP4.8h, TMP8.8h

    // stage 2 sum
    add TMP1.8h, TMP1.8h, TMP3.8h
    add TMP2.8h, TMP2.8h, TMP4.8h
    add v0.8h, TMP1.8h, TMP2.8h

    addv    h0, v0.8h
    fmov    w0, s0
    normalize_4
    ret

    #undef src
    #undef src_stride
    #undef dst
    #undef dst_stride

    #undef ROW1
    #undef TMP1
    #undef ROW2
    #undef TMP2
    #undef ROW3
    #undef TMP3
    #undef ROW4
    #undef TMP4
    #undef ROW5
    #undef TMP5
    #undef ROW6
    #undef TMP6
    #undef ROW7
    #undef TMP7
    #undef ROW8
    #undef TMP8
endfunc

function satd4x16_neon, export=1
    #define src         x0
    #define src_stride  x1
    #define dst         x2
    #define dst_stride  x3

    load_row    0, 1, src, dst, src_stride, dst_stride, 0
    load_row2   1, 2, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    2, 3, src, dst, src_stride, dst_stride, 0
    load_row2   3, 4, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    4, 5, src, dst, src_stride, dst_stride, 0
    load_row2   5, 6, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    6, 7, src, dst, src_stride, dst_stride, 0
    load_row2   7, 8, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    8, 9, src, dst, src_stride, dst_stride, 0
    load_row2   9, 10, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    10, 11, src, dst, src_stride, dst_stride, 0
    load_row2   11, 12, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    12, 13, src, dst, src_stride, dst_stride, 0
    load_row2   13, 14, src, dst, src_stride, dst_stride
    add         src, src, src_stride, lsl 1
    add         dst, dst, dst_stride, lsl 1

    load_row    14, 15, src, dst, src_stride, dst_stride, 0
    load_row2   15, 16, src, dst, src_stride, dst_stride

    // pack rows
    mov   v0.d[1], v2.d[0]
    mov   v1.d[1], v3.d[0]

    mov   v4.d[1], v6.d[0]
    mov   v5.d[1], v7.d[0]

    mov   v8.d[1], v10.d[0]
    mov   v9.d[1], v11.d[0]

    mov   v12.d[1], v14.d[0]
    mov   v13.d[1], v15.d[0]

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5
    butterfly v10, v11, v8, v9
    butterfly v14, v15, v12, v13

    interleave v0, v1, v2, v3
    interleave v4, v5, v6, v7
    interleave v8, v9, v10, v11
    interleave v12, v13, v14, v15

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5
    butterfly v10, v11, v8, v9
    butterfly v14, v15, v12, v13

    interleave_pairs v0, v1, v2, v3
    interleave_pairs v4, v5, v6, v7
    interleave_pairs v8, v9, v10, v11
    interleave_pairs v12, v13, v14, v15

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5
    butterfly v10, v11, v8, v9
    butterfly v14, v15, v12, v13

    interleave v0, v1, v2, v3
    interleave v4, v5, v6, v7
    interleave v8, v9, v10, v11
    interleave v12, v13, v14, v15

    butterfly v2, v3, v0, v1
    butterfly v6, v7, v4, v5
    butterfly v10, v11, v8, v9
    butterfly v14, v15, v12, v13

    abs  v2.8h, v2.8h
    abs  v3.8h, v3.8h
    abs  v6.8h, v6.8h
    abs  v7.8h, v7.8h
    abs  v10.8h, v10.8h
    abs  v11.8h, v11.8h
    abs  v14.8h, v14.8h
    abs  v15.8h, v15.8h

    add v2.8h, v2.8h, v3.8h
    add v6.8h, v6.8h, v7.8h
    add v10.8h, v10.8h, v11.8h
    add v14.8h, v14.8h, v15.8h

    add v2.8h, v2.8h, v6.8h
    add v10.8h, v10.8h, v14.8h
    add v0.8h, v2.8h, v10.8h

    addv    h0, v0.8h
    fmov    w0, s0
    normalize_4
    ret

    #undef src
    #undef src_stride
    #undef dst
    #undef dst_stride
endfunc
