/* Copyright (c) 2023, The rav1e contributors. All rights reserved
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

.macro LOAD_X4 t=q
    ldr     \t\()0, [x0]
    ldr     \t\()4, [x2]
    ldr     \t\()1, [x0, x1]
    ldr     \t\()5, [x2, x3]
    ldr     \t\()2, [x0, x11]
    ldr     \t\()6, [x2, x12]
    ldr     \t\()3, [x0, x8]
    ldr     \t\()7, [x2, x9]
    ldr     \t\()16, [x4]
    add     x0, x0, x1, lsl 2
    add     x2, x2, x3, lsl 2
    add     x4, x4, x5
    subs    w10, w10, #4
.ifc \t,q
    mov     v18.d[0], v16.d[1]
.else
    mov     v0.\t[1], v1.\t[0]
    mov     v4.\t[1], v5.\t[0]
    mov     v2.\t[1], v3.\t[0]
    mov     v6.\t[1], v7.\t[0]
.endif
.endm

.macro INIT width, height
    add     x11, x1, x1
    add     x12, x3, x3
    add     x8, x1, x1, lsl 1
    add     x9, x3, x3, lsl 1
    movi    v17.4s,  #0
    mov     w10, #(\height)
.endm

// x0: src: *const u8,
// x1: src_stride: isize,
// x2: dst: *const u8,
// x3: dst_stride: isize,
// x4: scale: *const u32,
// x5: scale_stride: isize,
function weighted_sse_4x4_neon, export=1
    INIT 4, 4
L(wsse_w4):
    LOAD_X4 t=s
    uabd    v0.8b, v0.8b, v4.8b  // diff pixel values
    uabd    v1.8b, v2.8b, v6.8b
    umull   v0.8h, v0.8b, v0.8b  // square
    umull   v1.8h, v1.8b, v1.8b
    uaddl   v2.4s, v0.4h, v1.4h  // accumulate
    uaddl2  v3.4s, v0.8h, v1.8h
    add     v0.4s, v2.4s, v3.4s
    addv    s0, v0.4s
    umull   v0.2d, v0.2s, v16.2s // apply scale
    urshr   d0, d0, #8
    add     v17.2d, v17.2d, v0.2d
    bne     L(wsse_w4)
    fmov    x0, d17
    ret
endfunc

.macro RET_SUM
    mov     v0.d[0], v17.d[1]
    add     d0, d0, d17
    fmov    x0, d0
    ret
.endm

function weighted_sse_8x8_neon, export=1
    INIT 8, 8
L(wsse_w8):
    LOAD_X4 t=d
    uabd    v0.16b, v0.16b, v4.16b  // diff pixel values
    uabd    v1.16b, v2.16b, v6.16b
    umull2  v2.8h, v0.16b, v0.16b   // square
    umull2  v3.8h, v1.16b, v1.16b
    umull   v0.8h, v0.8b, v0.8b
    umull   v1.8h, v1.8b, v1.8b
    uaddlp  v2.4s, v2.8h            // accumulate
    uaddlp  v3.4s, v3.8h
    uaddlp  v0.4s, v0.8h
    uaddlp  v1.4s, v1.8h
    uaddlp  v2.2d, v2.4s
    uadalp  v2.2d, v3.4s
    uadalp  v2.2d, v0.4s
    uadalp  v2.2d, v1.4s
    xtn     v0.2s, v2.2d
    umull   v0.2d, v0.2s, v16.2s    // apply scale
    urshr   v0.2d, v0.2d, #8
    add     v17.2d, v17.2d, v0.2d
    bne     L(wsse_w8)
    RET_SUM
endfunc

function weighted_sse_16x16_neon, export=1
    INIT 16, 16
L(wsse_w16):
    LOAD_X4
    uabd    v0.16b, v0.16b, v4.16b  // diff pixel values
    uabd    v1.16b, v1.16b, v5.16b
    uabd    v2.16b, v2.16b, v6.16b
    uabd    v3.16b, v3.16b, v7.16b
    umull2  v4.8h, v0.16b, v0.16b   // square
    umull2  v5.8h, v1.16b, v1.16b
    umull2  v6.8h, v2.16b, v2.16b
    umull2  v7.8h, v3.16b, v3.16b
    umull   v0.8h, v0.8b, v0.8b
    umull   v1.8h, v1.8b, v1.8b
    umull   v2.8h, v2.8b, v2.8b
    umull   v3.8h, v3.8b, v3.8b
    uaddlp  v4.4s, v4.8h            // accumulate
    uaddlp  v5.4s, v5.8h
    uaddlp  v6.4s, v6.8h
    uaddlp  v7.4s, v7.8h
    uaddlp  v0.4s, v0.8h
    uaddlp  v1.4s, v1.8h
    uaddlp  v2.4s, v2.8h
    uaddlp  v3.4s, v3.8h
    uaddlp  v4.2d, v4.4s
    uadalp  v4.2d, v5.4s
    uadalp  v4.2d, v6.4s
    uadalp  v4.2d, v7.4s
    xtn     v4.2s, v4.2d
    uaddlp  v0.2d, v0.4s
    uadalp  v0.2d, v1.4s
    uadalp  v0.2d, v2.4s
    uadalp  v0.2d, v3.4s
    xtn     v0.2s, v0.2d
    umull   v4.2d, v4.2s, v18.2s    // apply scale
    umull   v0.2d, v0.2s, v16.2s
    urshr   v4.2d, v4.2d, #8
    urshr   v0.2d, v0.2d, #8
    add     v17.2d, v17.2d, v4.2d
    add     v17.2d, v17.2d, v0.2d
    bne     L(wsse_w16)
    RET_SUM
endfunc

.macro weighted_sse width, height
function weighted_sse_\width\()x\height\()_neon, export=1
    INIT \width, \height
    b       L(wsse_w\width)
endfunc
.endm

weighted_sse 4, 8
weighted_sse 4, 16
weighted_sse 8, 4
weighted_sse 8, 16
weighted_sse 8, 32
weighted_sse 16, 4
weighted_sse 16, 8
weighted_sse 16, 32
weighted_sse 16, 64
