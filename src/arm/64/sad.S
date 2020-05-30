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
.if \width == 128
        movi            v3.4s,   #0
.else
        movi            v0.4s,   #0
.endif
        sxtw            x1,  w1
.if \width == 128
        movi            v18.4s,  #0
.endif
        sxtw            x3,  w3
        mov             w4,  \height
.if \width == 128
        mov             v2.16b,  v3.16b
.elseif \width >= 32
        mov             v1.16b,  v0.16b
.elseif \width == 16
        mov             v3.16b,  v0.16b
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
        subs            w4,  w4,  #1
        ldr             d1,  [x0]
        add             x0,  x0,  x1
        ldr             d2,  [x2]
        add             x2,  x2,  x3
        uabal           v0.8h,   v1.8b,   v2.8b
        bne             L(sad_w4)
        uaddlp          v0.2s,   v0.4h
        uaddlp          v0.1d,   v0.2s
        fmov            w0,  s0
        ret
endfunc

sad_rect 4, 8
sad_rect 4, 16

.macro horizontal_long_add_16x8
        dup             d2,  v1.d[0]
        dup             d1,  v1.d[1]
        uaddl           v1.4s,   v2.4h,   v1.4h
        dup             d2,  v0.d[0]
        dup             d0,  v0.d[1]
        uaddl           v0.4s,   v2.4h,   v0.4h
        add             v1.4s,   v1.4s,   v0.4s
        uaddlp          v1.2d,   v1.4s
        dup             d0,  v1.d[0]
        dup             d1,  v1.d[1]
        add             v1.2s,   v0.2s,   v1.2s
        umov            w0,  v1.s[0]
        ret
.endm

.macro horizontal_add_16x8
        uaddlp          v0.4s,   v0.8h
        uaddlp          v0.2d,   v0.4s
        dup             d1,  v0.d[0]
        dup             d0,  v0.d[1]
        add             v0.2s,   v1.2s,   v0.2s
        umov            w0,  v0.s[0]
        ret
.endm

function sad64x64_neon, export=1
        movi            v0.4s,   #0
        sxtw            x1,  w1
        sxtw            x3,  w3
        mov             w4,  #64
        mov             v1.16b,  v0.16b
L(sad_w64):
        ldr             q16, [x0]
        subs            w4,  w4,  #1
        ldr             q17, [x2]
        ldr             q6,  [x0, #16]
        ldr             q7,  [x2, #16]
        ldr             q4,  [x0, #32]
        ldr             q5,  [x2, #32]
        ldr             q2,  [x0, #48]
        add             x0,  x0,  x1
        ldr             q3,  [x2, #48]
        add             x2,  x2,  x3
        dup             d18, v16.d[0]
        dup             d19, v17.d[0]
        dup             d16, v16.d[1]
        dup             d17, v17.d[1]
        uabal           v0.8h,   v16.8b,  v17.8b
        dup             d16, v6.d[0]
        dup             d17, v7.d[0]
        dup             d6,  v6.d[1]
        dup             d7,  v7.d[1]
        uabal           v0.8h,   v6.8b,   v7.8b
        dup             d6,  v4.d[0]
        dup             d7,  v5.d[0]
        dup             d4,  v4.d[1]
        dup             d5,  v5.d[1]
        uabal           v1.8h,   v18.8b,  v19.8b
        uabal           v0.8h,   v4.8b,   v5.8b
        uabal           v1.8h,   v16.8b,  v17.8b
        dup             d4,  v2.d[0]
        dup             d5,  v3.d[0]
        uabal           v1.8h,   v6.8b,   v7.8b
        dup             d2,  v2.d[1]
        dup             d3,  v3.d[1]
        uabal           v1.8h,   v4.8b,   v5.8b
        uabal           v0.8h,   v2.8b,   v3.8b
        bne             L(sad_w64)
        horizontal_long_add_16x8
endfunc

sad_rect 64, 16
sad_rect 64, 32
sad_rect 64, 128

function sad128x128_neon, export=1
        movi            v3.4s,   #0
        sxtw            x1,  w1
        movi            v18.4s,  #0
        sxtw            x3,  w3
        mov             w4,  #128
        mov             v2.16b,  v3.16b
L(sad_w128):
        ldr             q0,  [x0]
        subs            w4,  w4,  #1
        ldr             q28, [x2]
        ldr             q25, [x0, #16]
        ldr             q26, [x2, #16]
        ldr             q23, [x0, #32]
        ldr             q24, [x2, #32]
        dup             d27, v0.d[0]
        ldr             q21, [x0, #48]
        ldr             q22, [x2, #48]
        dup             d29, v28.d[0]
        mov             v1.16b,  v18.16b
        dup             d28, v28.d[1]
        uabal           v1.8h,   v27.8b,  v29.8b
        dup             d27, v0.d[1]
        ldr             q19, [x0, #64]
        ldr             q20, [x2, #64]
        mov             v0.16b,  v18.16b
        uabal           v0.8h,   v27.8b,  v28.8b
        dup             d27, v25.d[0]
        dup             d28, v26.d[0]
        dup             d25, v25.d[1]
        dup             d26, v26.d[1]
        ldr             q16, [x0, #80]
        uabal           v0.8h,   v25.8b,  v26.8b
        ldr             q17, [x2, #80]
        uabal           v1.8h,   v27.8b,  v28.8b
        dup             d25, v23.d[0]
        dup             d26, v24.d[0]
        dup             d23, v23.d[1]
        dup             d24, v24.d[1]
        ldr             q6,  [x0, #96]
        uabal           v0.8h,   v23.8b,  v24.8b
        ldr             q7,  [x2, #96]
        uabal           v1.8h,   v25.8b,  v26.8b
        dup             d23, v21.d[0]
        dup             d24, v22.d[0]
        dup             d21, v21.d[1]
        dup             d22, v22.d[1]
        ldr             q4,  [x0, #112]
        uabal           v0.8h,   v21.8b,  v22.8b
        ldr             q5,  [x2, #112]
        uabal           v1.8h,   v23.8b,  v24.8b
        dup             d21, v19.d[0]
        add             x0,  x0,  x1
        dup             d22, v20.d[0]
        add             x2,  x2,  x3
        dup             d19, v19.d[1]
        dup             d20, v20.d[1]
        uabal           v0.8h,   v19.8b,  v20.8b
        dup             d19, v16.d[0]
        dup             d20, v17.d[0]
        dup             d16, v16.d[1]
        dup             d17, v17.d[1]
        uabal           v0.8h,   v16.8b,  v17.8b
        dup             d16, v6.d[0]
        dup             d17, v7.d[0]
        dup             d6,  v6.d[1]
        dup             d7,  v7.d[1]
        uabal           v1.8h,   v21.8b,  v22.8b
        uabal           v0.8h,   v6.8b,   v7.8b
        uabal           v1.8h,   v19.8b,  v20.8b
        dup             d6,  v4.d[0]
        dup             d7,  v5.d[0]
        uabal           v1.8h,   v16.8b,  v17.8b
        dup             d4,  v4.d[1]
        uabal           v1.8h,   v6.8b,   v7.8b
        dup             d5,  v5.d[1]
        uabal           v0.8h,   v4.8b,   v5.8b
        add             v0.8h,   v0.8h,   v1.8h
        dup             d1,  v0.d[0]
        dup             d0,  v0.d[1]
        uaddw           v2.4s,   v2.4s,   v1.4h
        uaddw           v3.4s,   v3.4s,   v0.4h
        bne             L(sad_w128)
        add             v2.4s,   v2.4s,   v3.4s
        uaddlp          v2.2d,   v2.4s
        dup             d0,  v2.d[1]
        add             v2.2s,   v0.2s,   v2.2s
        umov            w0,  v2.s[0]
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
        ldr             q4,  [x0]
        subs            w4,  w4,  #1
        ldr             q5,  [x2]
        ldr             q2,  [x0, #16]
        add             x0,  x0,  x1
        ldr             q3,  [x2, #16]
        add             x2,  x2,  x3
        dup             d6,  v4.d[0]
        dup             d7,  v5.d[0]
        dup             d4,  v4.d[1]
        dup             d5,  v5.d[1]
        uabal           v1.8h,   v6.8b,   v7.8b
        uabal           v0.8h,   v4.8b,   v5.8b
        dup             d4,  v2.d[0]
        dup             d5,  v3.d[0]
        dup             d2,  v2.d[1]
        dup             d3,  v3.d[1]
        uabal           v1.8h,   v4.8b,   v5.8b
        uabal           v0.8h,   v2.8b,   v3.8b
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
        mov             v3.16b,  v0.16b
L(sad_w16):
        ldr             q1,  [x0]
        subs            w4,  w4,  #1
        ldr             q2,  [x2]
        add             x0,  x0,  x1
        dup             d4,  v1.d[0]
        add             x2,  x2,  x3
        dup             d1,  v1.d[1]
        dup             d5,  v2.d[0]
        dup             d2,  v2.d[1]
        uabal           v3.8h,   v4.8b,   v5.8b
        uabal           v0.8h,   v1.8b,   v2.8b
        bne             L(sad_w16)
        add             v0.8h,   v0.8h,   v3.8h
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
        subs            w4,  w4,  #1
        ldr             d1,  [x0]
        add             x0,  x0,  x1
        ldr             d2,  [x2]
        add             x2,  x2,  x3
        uabal           v0.8h,   v1.8b,   v2.8b
        bne             L(sad_w8)
        horizontal_add_16x8
endfunc

sad_rect 8, 4
sad_rect 8, 16
sad_rect 8, 32
