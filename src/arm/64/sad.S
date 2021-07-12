/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
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
        ldr             d2,  [x0]
        ldr             d3,  [x2]
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
        ldp             q6,  q8,  [x0, #32]
        ldp             q7,  q9,  [x2, #32]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        uabal2          v1.8h,   v2.16b,  v3.16b
        uabal           v0.8h,   v4.8b,   v5.8b
        uabal2          v1.8h,   v4.16b,  v5.16b
        uabal           v0.8h,   v6.8b,   v7.8b
        uabal2          v1.8h,   v6.16b,  v7.16b
        uabal           v0.8h,   v8.8b,   v9.8b
        uabal2          v1.8h,   v8.16b,  v9.16b
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
        ldp             q6,  q8,  [x0, #32]
        ldp             q7,  q9,  [x2, #32]
        ldp             q10, q12, [x0, #64]
        ldp             q11, q13, [x2, #64]
        ldp             q14, q16, [x0, #96]
        ldp             q15, q17, [x2, #96]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabdl           v18.8h,  v2.8b,   v3.8b
        uabal2          v18.8h,  v2.16b,  v3.16b
        uabal           v18.8h,  v4.8b,   v5.8b
        uabal2          v18.8h,  v4.16b,  v5.16b
        uabal           v18.8h,  v6.8b,   v7.8b
        uabal2          v18.8h,  v6.16b,  v7.16b
        uabal           v18.8h,  v8.8b,   v9.8b
        uabal2          v18.8h,  v8.16b,  v9.16b
        uabal           v18.8h,  v10.8b,  v11.8b
        uabal2          v18.8h,  v10.16b, v11.16b
        uabal           v18.8h,  v12.8b,  v13.8b
        uabal2          v18.8h,  v12.16b, v13.16b
        uabal           v18.8h,  v14.8b,  v15.8b
        uabal2          v18.8h,  v14.16b, v15.16b
        uabal           v18.8h,  v16.8b,  v17.8b
        uabal2          v18.8h,  v16.16b, v17.16b
        uaddw           v0.4s,   v0.4s,   v18.4h
        uaddw2          v1.4s,   v1.4s,   v18.8h
        bne             L(sad_w128)
        add             v1.4s,   v1.4s,   v0.4s
        uaddlp          v1.2d,   v1.4s
        dup             d0,  v1.d[1]
        add             v1.2s,   v0.2s,   v1.2s
        umov            w0,  v1.s[0]
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
        ldr             q2,  [x0]
        ldr             q3,  [x2]
        ldr             q4,  [x0, #16]
        ldr             q5,  [x2, #16]
        add             x0,  x0,  x1
        add             x2,  x2,  x3
        subs            w4,  w4,  #1
        uabal           v0.8h,   v2.8b,   v3.8b
        uabal2          v1.8h,   v2.16b,  v3.16b
        uabal           v0.8h,   v4.8b,   v5.8b
        uabal2          v1.8h,   v4.16b,  v5.16b
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
