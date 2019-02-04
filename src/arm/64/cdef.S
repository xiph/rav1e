/*
 * Copyright © 2018, VideoLAN and dav1d authors
 * Copyright © 2019, Martin Storsjo
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "src/arm/asm.S"
#include "util.S"

.macro pad_top_bottom s1, s2, w, stride, rn, rw, ret
        tst             w6,  #1 // CDEF_HAVE_LEFT
        b.eq            2f
        // CDEF_HAVE_LEFT
        sub             \s1,  \s1,  #2
        sub             \s2,  \s2,  #2
        tst             w6,  #2 // CDEF_HAVE_RIGHT
        b.eq            1f
        // CDEF_HAVE_LEFT+CDEF_HAVE_RIGHT
        ldr             \rn\()0, [\s1]
        ldr             s1,      [\s1, #\w]
        ldr             \rn\()2, [\s2]
        ldr             s3,      [\s2, #\w]
        uxtl            v0.8h,   v0.8b
        uxtl            v1.8h,   v1.8b
        uxtl            v2.8h,   v2.8b
        uxtl            v3.8h,   v3.8b
        str             \rw\()0, [x0]
        str             d1,      [x0, #2*\w]
        add             x0,  x0,  #2*\stride
        str             \rw\()2, [x0]
        str             d3,      [x0, #2*\w]
.if \ret
        ret
.else
        add             x0,  x0,  #2*\stride
        b               3f
.endif

1:
        // CDEF_HAVE_LEFT+!CDEF_HAVE_RIGHT
        ldr             \rn\()0, [\s1]
        ldr             h1,      [\s1, #\w]
        ldr             \rn\()2, [\s2]
        ldr             h3,      [\s2, #\w]
        uxtl            v0.8h,   v0.8b
        uxtl            v1.8h,   v1.8b
        uxtl            v2.8h,   v2.8b
        uxtl            v3.8h,   v3.8b
        str             \rw\()0, [x0]
        str             s1,      [x0, #2*\w]
        str             s31,     [x0, #2*\w+4]
        add             x0,  x0,  #2*\stride
        str             \rw\()2, [x0]
        str             s3,      [x0, #2*\w]
        str             s31,     [x0, #2*\w+4]
.if \ret
        ret
.else
        add             x0,  x0,  #2*\stride
        b               3f
.endif

2:
        // !CDEF_HAVE_LEFT
        tst             w6,  #2 // CDEF_HAVE_RIGHT
        b.eq            1f
        // !CDEF_HAVE_LEFT+CDEF_HAVE_RIGHT
        ldr             \rn\()0, [\s1]
        ldr             h1,      [\s1, #\w]
        ldr             \rn\()2, [\s2]
        ldr             h3,      [\s2, #\w]
        uxtl            v0.8h,  v0.8b
        uxtl            v1.8h,  v1.8b
        uxtl            v2.8h,  v2.8b
        uxtl            v3.8h,  v3.8b
        str             s31, [x0]
        stur            \rw\()0, [x0, #4]
        str             s1,      [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        str             s31, [x0]
        stur            \rw\()2, [x0, #4]
        str             s3,      [x0, #4+2*\w]
.if \ret
        ret
.else
        add             x0,  x0,  #2*\stride
        b               3f
.endif

1:
        // !CDEF_HAVE_LEFT+!CDEF_HAVE_RIGHT
        ldr             \rn\()0, [\s1]
        ldr             \rn\()1, [\s2]
        uxtl            v0.8h,  v0.8b
        uxtl            v1.8h,  v1.8b
        str             s31,     [x0]
        stur            \rw\()0, [x0, #4]
        str             s31,     [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        str             s31,     [x0]
        stur            \rw\()1, [x0, #4]
        str             s31,     [x0, #4+2*\w]
.if \ret
        ret
.else
        add             x0,  x0,  #2*\stride
.endif
3:
.endm

// void dav1d_cdef_paddingX_neon(uint16_t *tmp, const pixel *src,
//                               ptrdiff_t src_stride, const pixel (*left)[2],
//                               /*const*/ pixel *const top[2], int h,
//                               enum CdefEdgeFlags edges);

.macro padding_func w, stride, rn, rw
function cdef_padding\w\()_neon, export=1
        movi            v30.16b, #255
        ushr            v30.8h, v30.8h, #1 // INT16_MAX
        mov             v31.16b, v30.16b
        sub             x0,  x0,  #2*(2*\stride+2)
        tst             w6,  #4 // CDEF_HAVE_TOP
        b.ne            1f
        // !CDEF_HAVE_TOP
        st1             {v30.8h, v31.8h}, [x0], #32
.if \w == 8
        st1             {v30.8h, v31.8h}, [x0], #32
.endif
        b               3f
1:
        // CDEF_HAVE_TOP
        ldr             x8,  [x4]
        ldr             x9,  [x4, #8]
        pad_top_bottom  x8,  x9, \w, \stride, \rn, \rw, 0

        // Middle section
3:
        tst             w6,  #1 // CDEF_HAVE_LEFT
        b.eq            2f
        // CDEF_HAVE_LEFT
        tst             w6,  #2 // CDEF_HAVE_RIGHT
        b.eq            1f
        // CDEF_HAVE_LEFT+CDEF_HAVE_RIGHT
0:
        ld1             {v0.h}[0], [x3], #2
        ldr             \rn\()1, [x1]
        ldr             h2,      [x1, #\w]
        add             x1,  x1,  x2
        subs            w5,  w5,  #1
        uxtl            v0.8h,  v0.8b
        uxtl            v1.8h,  v1.8b
        uxtl            v2.8h,  v2.8b
        str             s0,      [x0]
        stur            \rw\()1, [x0, #4]
        str             s2,      [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        b.gt            0b
        b               3f
1:
        // CDEF_HAVE_LEFT+!CDEF_HAVE_RIGHT
        ld1             {v0.h}[0], [x3], #2
.if \w == 8
        ld1             {v1.8b},   [x1], x2
.else
        ld1             {v1.s}[0], [x1], x2
.endif
        subs            w5,  w5,  #1
        uxtl            v0.8h,  v0.8b
        uxtl            v1.8h,  v1.8b
        str             s0,      [x0]
        stur            \rw\()1, [x0, #4]
        str             s31,     [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        b.gt            1b
        b               3f
2:
        tst             w6,  #2 // CDEF_HAVE_RIGHT
        b.eq            1f
        // !CDEF_HAVE_LEFT+CDEF_HAVE_RIGHT
0:
        ldr             \rn\()0, [x1]
        ldr             h1,      [x1, #\w]
        add             x1,  x1,  x2
        subs            w5,  w5,  #1
        uxtl            v0.8h,  v0.8b
        uxtl            v1.8h,  v1.8b
        str             s31,     [x0]
        stur            \rw\()0, [x0, #4]
        str             s1,      [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        b.gt            0b
        b               3f
1:
        // !CDEF_HAVE_LEFT+!CDEF_HAVE_RIGHT
.if \w == 8
        ld1             {v0.8b},   [x1], x2
.else
        ld1             {v0.s}[0], [x1], x2
.endif
        subs            w5,  w5,  #1
        uxtl            v0.8h,  v0.8b
        str             s31,     [x0]
        stur            \rw\()0, [x0, #4]
        str             s31,     [x0, #4+2*\w]
        add             x0,  x0,  #2*\stride
        b.gt            1b

3:
        tst             w6,  #8 // CDEF_HAVE_BOTTOM
        b.ne            1f
        // !CDEF_HAVE_BOTTOM
        st1             {v30.8h, v31.8h}, [x0], #32
.if \w == 8
        st1             {v30.8h, v31.8h}, [x0], #32
.endif
        ret
1:
        // CDEF_HAVE_BOTTOM
        add             x9,  x1,  x2
        pad_top_bottom  x1,  x9, \w, \stride, \rn, \rw, 1
endfunc
.endm

padding_func 8, 16, d, q
padding_func 4, 8,  s, d

.macro dir_table w, stride
const directions\w
        .byte           -1 * \stride + 1, -2 * \stride + 2
        .byte            0 * \stride + 1, -1 * \stride + 2
        .byte            0 * \stride + 1,  0 * \stride + 2
        .byte            0 * \stride + 1,  1 * \stride + 2
        .byte            1 * \stride + 1,  2 * \stride + 2
        .byte            1 * \stride + 0,  2 * \stride + 1
        .byte            1 * \stride + 0,  2 * \stride + 0
        .byte            1 * \stride + 0,  2 * \stride - 1
// Repeated, to avoid & 7
        .byte           -1 * \stride + 1, -2 * \stride + 2
        .byte            0 * \stride + 1, -1 * \stride + 2
        .byte            0 * \stride + 1,  0 * \stride + 2
        .byte            0 * \stride + 1,  1 * \stride + 2
        .byte            1 * \stride + 1,  2 * \stride + 2
        .byte            1 * \stride + 0,  2 * \stride + 1
endconst
.endm

dir_table 8, 16
dir_table 4, 8

const pri_taps
        .byte           4, 2, 3, 3
endconst

.macro load_px d1, d2, w
.if \w == 8
        add             x6,  x2,  w9, sxtb #1       // x + off
        sub             x9,  x2,  w9, sxtb #1       // x - off
        ld1             {\d1\().8h}, [x6]           // p0
        ld1             {\d2\().8h}, [x9]           // p1
.else
        add             x6,  x2,  w9, sxtb #1       // x + off
        sub             x9,  x2,  w9, sxtb #1       // x - off
        ld1             {\d1\().4h}, [x6]           // p0
        add             x6,  x6,  #2*8              // += stride
        ld1             {\d2\().4h}, [x9]           // p1
        add             x9,  x9,  #2*8              // += stride
        ld1             {\d1\().d}[1], [x6]         // p0
        ld1             {\d2\().d}[1], [x9]         // p1
.endif
.endm
.macro handle_pixel s1, s2, threshold, thresh_vec, shift, tap
        cmeq            v16.8h,  \s1\().8h,  v31.8h
        cmeq            v17.8h,  \s2\().8h,  v31.8h
        bic             v16.16b, \s1\().16b, v16.16b
        bic             v17.16b, \s2\().16b, v17.16b
        umin            v2.8h,   v2.8h,  \s1\().8h
        umax            v3.8h,   v3.8h,  v16.8h
        umin            v2.8h,   v2.8h,  \s2\().8h
        umax            v3.8h,   v3.8h,  v17.8h

        cbz             \threshold, 3f
        uabd            v16.8h, v0.8h,  \s1\().8h   // abs(diff)
        uabd            v20.8h, v0.8h,  \s2\().8h   // abs(diff)
        ushl            v17.8h, v16.8h, \shift      // abs(diff) >> shift
        ushl            v21.8h, v20.8h, \shift      // abs(diff) >> shift
        sub             v17.8h, \thresh_vec, v17.8h // threshold - (abs(diff) >> shift)
        sub             v21.8h, \thresh_vec, v21.8h // threshold - (abs(diff) >> shift)
        smax            v17.8h, v29.8h, v17.8h      // imax(0, threshold - ())
        smax            v21.8h, v29.8h, v21.8h      // imax(0, threshold - ())
        cmhi            v18.8h, v0.8h,  \s1\().8h   // px > p0
        cmhi            v22.8h, v0.8h,  \s2\().8h   // px > p1
        smin            v17.8h, v17.8h, v16.8h      // imin(abs(diff), imax())
        smin            v21.8h, v21.8h, v20.8h      // imin(abs(diff), imax())
        dup             v19.8h, \tap                // taps[k]/taps[k]
        neg             v16.8h, v17.8h              // -imin()
        neg             v20.8h, v21.8h              // -imin()
        bsl             v18.16b, v16.16b, v17.16b   // constrain() = apply_sign()
        bsl             v22.16b, v20.16b, v21.16b   // constrain() = apply_sign()
        mla             v1.8h,  v18.8h, v19.8h      // sum += taps[k] * constrain()
        mla             v1.8h,  v22.8h, v19.8h      // sum += taps[k] * constrain()
3:
.endm

// void dav1d_cdef_filterX_neon(pixel *dst, ptrdiff_t dst_stride,
//                              const uint16_t *tmp, int pri_strength,
//                              int sec_strength, int dir, int damping, int h);
.macro filter w
function cdef_filter\w\()_neon, export=1
        movrel          x8,  pri_taps
        and             w9,  w3,  #1
        add             x8,  x8,  w9, uxtw #1
        movrel          x9,  directions\w
        add             x5,  x9,  w5, uxtw #1
        movi            v31.16b,  #255
        movi            v30.8h,   #15
        movi            v29.8h,   #0
        dup             v28.8h,   w6                // damping
        ushr            v31.8h,   v31.8h, #1        // INT16_MAX

        dup             v25.8h, w3                  // threshold
        dup             v27.8h, w4                  // threshold
        clz             v24.8h, v25.8h              // clz(threshold)
        clz             v26.8h, v27.8h              // clz(threshold)
        sub             v24.8h, v30.8h, v24.8h      // ulog2(threshold)
        sub             v26.8h, v30.8h, v26.8h      // ulog2(threshold)
        sub             v24.8h, v28.8h, v24.8h      // damping - ulog2(threshold)
        sub             v26.8h, v28.8h, v26.8h      // damping - ulog2(threshold)
        smax            v24.8h, v29.8h, v24.8h      // shift = imax(0, damping - ulog2(threshold))
        smax            v26.8h, v29.8h, v26.8h      // shift = imax(0, damping - ulog2(threshold))
        neg             v24.8h, v24.8h              // -shift
        neg             v26.8h, v26.8h              // -shift

1:
.if \w == 8
        ld1             {v0.8h}, [x2]               // px
.else
        add             x12, x2,  #2*8
        ld1             {v0.4h},   [x2]             // px
        ld1             {v0.d}[1], [x12]            // px
.endif

        movi            v1.8h,  #0                  // sum
        mov             v2.16b, v0.16b              // min
        mov             v3.16b, v0.16b              // max

        // Instead of loading sec_taps 2, 1 from memory, just set it
        // to 2 initially and decrease for the second round.
        mov             w11, #2                     // sec_taps[0]

2:
        ldrb            w9,  [x5]                   // off1

        load_px         v4,  v5, \w

        add             x5,  x5,  #4                // +2*2
        ldrb            w9,  [x5]                   // off2
        load_px         v6,  v7,  \w

        ldrb            w10, [x8]                   // *pri_taps

        handle_pixel    v4,  v5,  w3,  v25.8h, v24.8h, w10

        add             x5,  x5,  #8                // +2*4
        ldrb            w9,  [x5]                   // off3
        load_px         v4,  v5,  \w

        handle_pixel    v6,  v7,  w4,  v27.8h, v26.8h, w11

        handle_pixel    v4,  v5,  w4,  v27.8h, v26.8h, w11

        sub             x5,  x5,  #11               // x8 -= 2*(2+4); x8 += 1;
        subs            w11, w11, #1                // sec_tap-- (value)
        add             x8,  x8,  #1                // pri_taps++ (pointer)
        b.ne            2b

        sshr            v4.8h,  v1.8h,  #15         // -(sum < 0)
        add             v1.8h,  v1.8h,  v4.8h       // sum - (sum < 0)
        srshr           v1.8h,  v1.8h,  #4          // (8 + sum - (sum < 0)) >> 4
        add             v0.8h,  v0.8h,  v1.8h       // px + (8 + sum ...) >> 4
        smin            v0.8h,  v0.8h,  v3.8h
        smax            v0.8h,  v0.8h,  v2.8h       // iclip(px + .., min, max)
        xtn             v0.8b,  v0.8h
.if \w == 8
        add             x2,  x2,  #2*16             // tmp += tmp_stride
        subs            w7,  w7,  #1                // h--
        st1             {v0.8b}, [x0], x1
.else
        st1             {v0.s}[0], [x0], x1
        add             x2,  x2,  #2*16             // tmp += 2*tmp_stride
        subs            w7,  w7,  #2                // h -= 2
        st1             {v0.s}[1], [x0], x1
.endif

        // Reset pri_taps/sec_taps back to the original point
        sub             x5,  x5,  #2
        sub             x8,  x8,  #2

        b.gt            1b
        ret
endfunc
.endm

filter 8
filter 4
