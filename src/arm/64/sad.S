/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 * Copyright (c) 2020-2023, The rav1e contributors. All rights reserved
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

.macro sad_rect width, height
function sad\width\()x\height\()_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  \height
.if \width >= 16
        mov             v1.16b,  v0.16b
.endif
        b               L(sad_w\width\())
endfunc
.endm

function sad4x4_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #4
L(sad_w4):
        ldr             s2,  [x0]
        ldr             s3,  [x2]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        bne             L(sad_w4)
        uaddlp          v0.2s,   v0.4h
        uaddlp          v0.1d,   v0.2s
        fmov            w0,  s0
        ret
endfunc

sad_rect 4, 8
sad_rect 4, 16

.macro horizontal_long_add_16x8
        ushll           v2.4s,   v1.4h,   #0
        uaddw2          v1.4s,   v2.4s,   v1.8h
        uaddw           v1.4s,   v1.4s,   v0.4h
        uaddw2          v0.4s,   v1.4s,   v0.8h
        uaddlp          v0.2d,   v0.4s
        ext             v1.16b,  v0.16b,  v0.16b,  #8
        add             v0.2s,   v1.2s,   v0.2s
        fmov            w0,  s0
        ret
.endm

.macro horizontal_add_16x8
        uaddlp          v0.4s,   v0.8h
        uaddlp          v0.2d,   v0.4s
        ext             v1.16b,  v0.16b,  v0.16b,  #8
        add             v0.2s,   v1.2s,   v0.2s
        fmov            w0,  s0
        ret
.endm

function sad64x64_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #64
        mov             v1.16b,  v0.16b
L(sad_w64):
        ldp             q2,  q4,  [x0]
        ldp             q3,  q5,  [x2]
        ldp             q6,  q16, [x0, #32]
        ldp             q7,  q17, [x2, #32]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        uabal2          v1.8h,   v2.16b,  v3.16b
        uabal           v0.8h,   v4.8b,   v5.8b
        uabal2          v1.8h,   v4.16b,  v5.16b
        uabal           v0.8h,   v6.8b,   v7.8b
        uabal2          v1.8h,   v6.16b,  v7.16b
        uabal           v0.8h,   v16.8b,  v17.8b
        uabal2          v1.8h,   v16.16b, v17.16b
        bne             L(sad_w64)
        horizontal_long_add_16x8
endfunc

sad_rect 64, 16
sad_rect 64, 32
sad_rect 64, 128

function sad128x128_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #128
        mov             v1.16b,  v0.16b
L(sad_w128):
        ldp             q2,  q4,  [x0]
        ldp             q3,  q5,  [x2]
        ldp             q6,  q16, [x0, #32]
        ldp             q7,  q17, [x2, #32]
        ldp             q18, q20, [x0, #64]
        ldp             q19, q21, [x2, #64]
        ldp             q22, q24, [x0, #96]
        ldp             q23, q25, [x2, #96]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabdl           v26.8h,   v2.8b,   v3.8b
        uabal2          v26.8h,   v2.16b,  v3.16b
        uabal           v26.8h,   v4.8b,   v5.8b
        uabal2          v26.8h,   v4.16b,  v5.16b
        uabal           v26.8h,   v6.8b,   v7.8b
        uabal2          v26.8h,   v6.16b,  v7.16b
        uabal           v26.8h,   v16.8b,  v17.8b
        uabal2          v26.8h,   v16.16b, v17.16b
        uabal           v26.8h,   v18.8b,  v19.8b
        uabal2          v26.8h,   v18.16b, v19.16b
        uabal           v26.8h,   v20.8b,  v21.8b
        uabal2          v26.8h,   v20.16b, v21.16b
        uabal           v26.8h,   v22.8b,  v23.8b
        uabal2          v26.8h,   v22.16b, v23.16b
        uabal           v26.8h,   v24.8b,  v25.8b
        uabal2          v26.8h,   v24.16b, v25.16b
        uaddw           v1.4s,   v1.4s,   v26.4h
        uaddw2          v0.4s,   v0.4s,   v26.8h
        bne             L(sad_w128)
        add             v0.4s,   v0.4s,   v1.4s
        uaddlp          v0.2d,   v0.4s
        dup             d3,  v0.d[1]
        add             v0.2s,   v0.2s,   v3.2s
        umov            w0,  v0.s[0]
        ret
endfunc

sad_rect 128, 64

function sad32x32_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #32
        mov             v1.16b,  v0.16b
L(sad_w32):
        ldp             q2,  q4,  [x0]
        ldp             q3,  q5,  [x2]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v1.8h,   v2.8b,   v3.8b
        uabal2          v0.8h,   v2.16b,  v3.16b
        uabal           v1.8h,   v4.8b,   v5.8b
        uabal2          v0.8h,   v4.16b,  v5.16b
        bne             L(sad_w32)
        add             v0.8h,   v0.8h,   v1.8h
        horizontal_add_16x8
endfunc

sad_rect 32, 8
sad_rect 32, 16
sad_rect 32, 64

function sad16x16_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #16
        mov             v1.16b,  v0.16b
L(sad_w16):
        ldr             q2,  [x0]
        ldr             q3,  [x2]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        uabal2          v1.8h,   v2.16b,  v3.16b
        bne             L(sad_w16)
        add             v0.8h,   v0.8h,   v1.8h
        horizontal_add_16x8
endfunc

sad_rect 16, 4
sad_rect 16, 8
sad_rect 16, 32
sad_rect 16, 64

function sad8x8_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #8
L(sad_w8):
        ldr             d2,  [x0]
        ldr             d3,  [x2]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        bne             L(sad_w8)
        horizontal_add_16x8
endfunc

sad_rect 8, 4
sad_rect 8, 16
sad_rect 8, 32
