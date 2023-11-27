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

// v0: tmp register
// v1: src input
// v2: dst input
// v3 = sum(src_{i,j})
// v4 = sum(src_{i,j}^2)
// v5 = sum(dst_{i,j})
// v6 = sum(dst_{i,j}^2)
// v7 = sum(src_{i,j} * dst_{i,j})
// v16: zero register
.macro CDEF_DIST_W8
    uabal   v3.8h, v1.8b, v16.8b // sum pixel values
    umull   v0.8h, v1.8b, v1.8b  // square
    uabal   v4.4s, v0.4h, v16.4h // accumulate
    uabal2  v4.4s, v0.8h, v16.8h
    uabal   v5.8h, v2.8b, v16.8b // same as above, but for dst
    umull   v0.8h, v2.8b, v2.8b
    uabal   v6.4s, v0.4h, v16.4h
    uabal2  v6.4s, v0.8h, v16.8h
    umull   v0.8h, v1.8b, v2.8b // src_{i,j} * dst_{i,j}
    uabal   v7.4s, v0.4h, v16.4h
    uabal2  v7.4s, v0.8h, v16.8h
.endm

.macro CDEF_DIST_REFINE shift=0
    addv    h3, v3.8h
    umull   v3.4s, v3.4h, v3.4h
    urshr   v3.4s, v3.4s, #(6-\shift) // s3: sum(src_{i,j})^2 / N
    addv    s4, v4.4s                 // s4: sum(src_{i,j}^2)
    addv    h5, v5.8h
    umull   v5.4s, v5.4h, v5.4h
    urshr   v5.4s, v5.4s, #(6-\shift) // s5: sum(dst_{i,j})^2 / N
    addv    s6, v6.4s                 // s6: sum(dst_{i,j}^2)
    addv    s7, v7.4s
    add     v0.4s, v4.4s, v6.4s
    sub     v0.4s, v0.4s, v7.4s
    sub     v0.4s, v0.4s, v7.4s       // s0: sse
    uqsub   v4.4s, v4.4s, v3.4s       // s4: svar
    uqsub   v6.4s, v6.4s, v5.4s       // s6: dvar
.if \shift != 0
    shl     v4.4s, v4.4s, #\shift
    shl     v6.4s, v6.4s, #\shift
.endif
    str     s4, [x4]
    str     s6, [x4, #4]
    str     s0, [x4, #8]
.endm

.macro LOAD_ROW
    ldr     q1, [x0]
    ldr     q2, [x2]
    add     x0, x0, x1
    add     x2, x2, x3
.endm

.macro LOAD_ROWS
    ldr     s1, [x0]
    ldr     s2, [x2]
    ldr     s0, [x0, x1]
    ldr     s17, [x2, x3]
    add     x0, x0, x1, lsl 1
    add     x2, x2, x3, lsl 1
    zip1    v1.2s, v1.2s, v0.2s
    zip1    v2.2s, v2.2s, v17.2s
.endm

.macro CDEF_DIST_INIT width, height
.irp i, v3.8h, v4.8h, v5.8h, v6.8h, v7.8h, v16.8h
    movi    \i,  #0
.endr
.if \width == 4
    mov     w5, #(\height / 2)
.else
    mov     w5, #\height
.endif
.endm

// x0: src: *const u8,
// x1: src_stride: isize,
// x2: dst: *const u8,
// x3: dst_stride: isize,
// x4: ret_ptr: *mut u32,
function cdef_dist_kernel_4x4_neon, export=1
    CDEF_DIST_INIT 4, 4
L(cdk_4x4):
    LOAD_ROWS
    CDEF_DIST_W8
    subs w5, w5, #1
    bne L(cdk_4x4)
    CDEF_DIST_REFINE 2
    ret
endfunc

function cdef_dist_kernel_4x8_neon, export=1
    CDEF_DIST_INIT 4, 8
L(cdk_4x8):
    LOAD_ROWS
    CDEF_DIST_W8
    subs w5, w5, #1
    bne L(cdk_4x8)
    CDEF_DIST_REFINE 1
    ret
endfunc

function cdef_dist_kernel_8x4_neon, export=1
    CDEF_DIST_INIT 8, 4
L(cdk_8x4):
    LOAD_ROW
    CDEF_DIST_W8
    subs w5, w5, #1
    bne L(cdk_8x4)
    CDEF_DIST_REFINE 1
    ret
endfunc

function cdef_dist_kernel_8x8_neon, export=1
    CDEF_DIST_INIT 8, 8
L(cdk_8x8):
    LOAD_ROW
    CDEF_DIST_W8
    subs w5, w5, #1
    bne L(cdk_8x8)
    CDEF_DIST_REFINE
    ret
endfunc
