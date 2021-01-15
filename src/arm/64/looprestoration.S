/*
 * Copyright © 2018, VideoLAN and dav1d authors
 * Copyright © 2018, Martin Storsjo
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

const right_ext_mask_buf
        .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
right_ext_mask:
        .byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
        .byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
        .byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
        .byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
endconst

// void dav1d_wiener_filter_h_8bpc_neon(int16_t *dst, const pixel (*left)[4],
//                                      const pixel *src, ptrdiff_t stride,
//                                      const int16_t fh[8], intptr_t w,
//                                      int h, enum LrEdgeFlags edges);
function wiener_filter_h_8bpc_neon, export=1
        mov             w8,  w5
        ld1             {v0.8h},  [x4]
        mov             w9,  #(1 << 14) - (1 << 2)
        dup             v30.8h,  w9
        movi            v31.8h,  #8, lsl #8
        // Calculate mid_stride
        add             w10, w5,  #7
        bic             w10, w10, #7
        lsl             w10, w10, #1

        // Set up pointers for reading/writing alternate rows
        add             x12, x0,  x10
        lsl             w10, w10, #1
        add             x13, x2,  x3
        lsl             x3,  x3,  #1

        // Subtract the aligned width from mid_stride
        add             w11, w5,  #7
        bic             w11, w11, #7
        sub             x10, x10, w11, uxtw #1

        // Subtract the number of pixels read from the source stride
        add             w11, w11, #8
        sub             x3,  x3,  w11, uxtw

        // Set up the src pointers to include the left edge, for LR_HAVE_LEFT, left == NULL
        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            2f
        // LR_HAVE_LEFT
        cbnz            x1,  0f
        // left == NULL
        sub             x2,  x2,  #3
        sub             x13, x13, #3
        b               1f
0:      // LR_HAVE_LEFT, left != NULL
2:      // !LR_HAVE_LEFT, increase the stride.
        // For this case we don't read the left 3 pixels from the src pointer,
        // but shift it as if we had done that.
        add             x3,  x3,  #3


1:      // Loop vertically
        ld1             {v3.16b},  [x2],  #16
        ld1             {v5.16b},  [x13], #16

        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            0f
        cbz             x1,  2f
        // LR_HAVE_LEFT, left != NULL
        ld1             {v2.s}[3],  [x1], #4
        // Move x2/x13 back to account for the last 3 bytes we loaded earlier,
        // which we'll shift out.
        sub             x2,  x2,  #3
        sub             x13, x13, #3
        ld1             {v4.s}[3],  [x1], #4
        ext             v3.16b, v2.16b, v3.16b, #13
        ext             v5.16b, v4.16b, v5.16b, #13
        b               2f
0:
        // !LR_HAVE_LEFT, fill v2 with the leftmost byte
        // and shift v3 to have 3x the first byte at the front.
        dup             v2.16b, v3.b[0]
        dup             v4.16b, v5.b[0]
        // Move x2 back to account for the last 3 bytes we loaded before,
        // which we shifted out.
        sub             x2,  x2,  #3
        sub             x13, x13, #3
        ext             v3.16b, v2.16b, v3.16b, #13
        ext             v5.16b, v4.16b, v5.16b, #13

2:
        uxtl            v2.8h,  v3.8b
        uxtl2           v3.8h,  v3.16b
        uxtl            v4.8h,  v5.8b
        uxtl2           v5.8h,  v5.16b

        tst             w7,  #2 // LR_HAVE_RIGHT
        b.ne            4f
        // If we'll need to pad the right edge, load that byte to pad with
        // here since we can find it pretty easily from here.
        sub             w9,  w5, #14
        ldr             b28, [x2,  w9, sxtw]
        ldr             b29, [x13, w9, sxtw]
        // Fill v28/v29 with the right padding pixel
        dup             v28.8h,  v28.h[0]
        dup             v29.8h,  v29.h[0]
3:      // !LR_HAVE_RIGHT

        // Check whether we need to pad the right edge
        cmp             w5,  #11
        b.ge            4f   // If w >= 11, all used input pixels are valid

        // 1 <= w < 11, w+3 pixels valid in v2-v3. For w=9 or w=10,
        // this ends up called again; it's not strictly needed in those
        // cases (we pad enough here), but keeping the code as simple as possible.

        // Insert padding in v2/3.h[w+3] onwards; fuse the +3 (*2) into the
        // buffer pointer.
        movrel          x4,  right_ext_mask, -6
        sub             x4,  x4,  w5,  uxtw #1
        ld1             {v26.16b, v27.16b}, [x4]

        bit             v2.16b,  v28.16b, v26.16b
        bit             v3.16b,  v28.16b, v27.16b
        bit             v4.16b,  v29.16b, v26.16b
        bit             v5.16b,  v29.16b, v27.16b

4:      // Loop horizontally
        // Interleaving the mul/mla chains actually hurts performance
        // significantly on Cortex A53, thus keeping mul/mla tightly
        // chained like this.
        ext             v17.16b, v2.16b,  v3.16b, #4
        ext             v19.16b, v2.16b,  v3.16b, #8
        ext             v16.16b, v2.16b,  v3.16b, #2
        ext             v20.16b, v2.16b,  v3.16b, #10
        ext             v21.16b, v2.16b,  v3.16b, #12
        ext             v18.16b, v2.16b,  v3.16b, #6
        add             v19.8h,  v19.8h,  v17.8h
        add             v20.8h,  v20.8h,  v16.8h
        add             v21.8h,  v21.8h,  v2.8h
        mul             v6.8h,   v18.8h,  v0.h[3]
        mla             v6.8h,   v19.8h,  v0.h[4]
        mla             v6.8h,   v20.8h,  v0.h[5]
        mla             v6.8h,   v21.8h,  v0.h[6]
        ext             v23.16b, v4.16b,  v5.16b, #4
        ext             v25.16b, v4.16b,  v5.16b, #8
        ext             v22.16b, v4.16b,  v5.16b, #2
        ext             v26.16b, v4.16b,  v5.16b, #10
        ext             v27.16b, v4.16b,  v5.16b, #12
        ext             v24.16b, v4.16b,  v5.16b, #6
        add             v25.8h,  v25.8h,  v23.8h
        add             v26.8h,  v26.8h,  v22.8h
        add             v27.8h,  v27.8h,  v4.8h
        mul             v7.8h,   v24.8h,  v0.h[3]
        mla             v7.8h,   v25.8h,  v0.h[4]
        mla             v7.8h,   v26.8h,  v0.h[5]
        mla             v7.8h,   v27.8h,  v0.h[6]

        shl             v18.8h,  v18.8h,  #7
        shl             v24.8h,  v24.8h,  #7
        sub             v18.8h,  v18.8h,  v30.8h
        sub             v24.8h,  v24.8h,  v30.8h
        sqadd           v6.8h,   v6.8h,   v18.8h
        sqadd           v7.8h,   v7.8h,   v24.8h
        sshr            v6.8h,   v6.8h,   #3
        sshr            v7.8h,   v7.8h,   #3
        add             v6.8h,   v6.8h,   v31.8h
        add             v7.8h,   v7.8h,   v31.8h

        subs            w5,  w5,  #8

        st1             {v6.8h},  [x0],  #16
        st1             {v7.8h},  [x12], #16

        b.le            9f
        tst             w7,  #2 // LR_HAVE_RIGHT
        mov             v2.16b,  v3.16b
        mov             v4.16b,  v5.16b
        ld1             {v3.8b},  [x2],  #8
        ld1             {v5.8b},  [x13], #8
        uxtl            v3.8h,   v3.8b
        uxtl            v5.8h,   v5.8b
        b.ne            4b // If we don't need to pad, just keep filtering.
        b               3b // If we need to pad, check how many pixels we have left.

9:
        subs            w6,  w6,  #2
        b.le            0f
        // Jump to the next row and loop horizontally
        add             x0,  x0,  x10
        add             x12, x12, x10
        add             x2,  x2,  x3
        add             x13, x13, x3
        mov             w5,  w8
        b               1b
0:
        ret
endfunc

// void dav1d_wiener_filter_v_8bpc_neon(pixel *dst, ptrdiff_t stride,
//                                      const int16_t *mid, int w, int h,
//                                      const int16_t fv[8], enum LrEdgeFlags edges,
//                                      ptrdiff_t mid_stride);
function wiener_filter_v_8bpc_neon, export=1
        mov             w8,  w4
        ld1             {v0.8h},  [x5]

        // Calculate the number of rows to move back when looping vertically
        mov             w11, w4
        tst             w6,  #4 // LR_HAVE_TOP
        b.eq            0f
        sub             x2,  x2,  x7,  lsl #1
        add             w11, w11, #2
0:
        tst             w6,  #8 // LR_HAVE_BOTTOM
        b.eq            1f
        add             w11, w11, #2

1:      // Start of horizontal loop; start one vertical filter slice.
        // Load rows into v16-v19 and pad properly.
        tst             w6,  #4 // LR_HAVE_TOP
        ld1             {v16.8h}, [x2], x7
        b.eq            2f
        // LR_HAVE_TOP
        ld1             {v18.8h}, [x2], x7
        mov             v17.16b, v16.16b
        ld1             {v19.8h}, [x2], x7
        b               3f
2:      // !LR_HAVE_TOP
        mov             v17.16b, v16.16b
        mov             v18.16b, v16.16b
        mov             v19.16b, v16.16b

3:
        cmp             w4,  #4
        b.lt            5f
        // Start filtering normally; fill in v20-v22 with unique rows.
        ld1             {v20.8h}, [x2], x7
        ld1             {v21.8h}, [x2], x7
        ld1             {v22.8h}, [x2], x7

4:
.macro filter compare
        subs            w4,  w4,  #1
        // Interleaving the mul/mla chains actually hurts performance
        // significantly on Cortex A53, thus keeping mul/mla tightly
        // chained like this.
        add             v25.8h, v18.8h,  v20.8h
        add             v26.8h, v17.8h,  v21.8h
        add             v27.8h, v16.8h,  v22.8h
        smull           v2.4s,  v19.4h,  v0.h[3]
        smlal           v2.4s,  v25.4h,  v0.h[4]
        smlal           v2.4s,  v26.4h,  v0.h[5]
        smlal           v2.4s,  v27.4h,  v0.h[6]
        smull2          v3.4s,  v19.8h,  v0.h[3]
        smlal2          v3.4s,  v25.8h,  v0.h[4]
        smlal2          v3.4s,  v26.8h,  v0.h[5]
        smlal2          v3.4s,  v27.8h,  v0.h[6]
        sqrshrun        v2.4h,  v2.4s,   #11
        sqrshrun2       v2.8h,  v3.4s,   #11
        sqxtun          v2.8b,  v2.8h
        st1             {v2.8b}, [x0], x1
.if \compare
        cmp             w4,  #4
.else
        b.le            9f
.endif
        mov             v16.16b,  v17.16b
        mov             v17.16b,  v18.16b
        mov             v18.16b,  v19.16b
        mov             v19.16b,  v20.16b
        mov             v20.16b,  v21.16b
        mov             v21.16b,  v22.16b
.endm
        filter          1
        b.lt            7f
        ld1             {v22.8h}, [x2], x7
        b               4b

5:      // Less than 4 rows in total; not all of v20-v21 are filled yet.
        tst             w6,  #8 // LR_HAVE_BOTTOM
        b.eq            6f
        // LR_HAVE_BOTTOM
        cmp             w4,  #2
        // We load at least 2 rows in all cases.
        ld1             {v20.8h}, [x2], x7
        ld1             {v21.8h}, [x2], x7
        b.gt            53f // 3 rows in total
        b.eq            52f // 2 rows in total
51:     // 1 row in total, v19 already loaded, load edge into v20-v22.
        mov             v22.16b,  v21.16b
        b               8f
52:     // 2 rows in total, v19 already loaded, load v20 with content data
        // and 2 rows of edge.
        ld1             {v22.8h}, [x2], x7
        mov             v23.16b,  v22.16b
        b               8f
53:
        // 3 rows in total, v19 already loaded, load v20 and v21 with content
        // and 2 rows of edge.
        ld1             {v22.8h}, [x2], x7
        ld1             {v23.8h}, [x2], x7
        mov             v24.16b,  v23.16b
        b               8f

6:
        // !LR_HAVE_BOTTOM
        cmp             w4,  #2
        b.gt            63f // 3 rows in total
        b.eq            62f // 2 rows in total
61:     // 1 row in total, v19 already loaded, pad that into v20-v22.
        mov             v20.16b,  v19.16b
        mov             v21.16b,  v19.16b
        mov             v22.16b,  v19.16b
        b               8f
62:     // 2 rows in total, v19 already loaded, load v20 and pad that into v21-v23.
        ld1             {v20.8h}, [x2], x7
        mov             v21.16b,  v20.16b
        mov             v22.16b,  v20.16b
        mov             v23.16b,  v20.16b
        b               8f
63:
        // 3 rows in total, v19 already loaded, load v20 and v21 and pad v21 into v22-v24.
        ld1             {v20.8h}, [x2], x7
        ld1             {v21.8h}, [x2], x7
        mov             v22.16b,  v21.16b
        mov             v23.16b,  v21.16b
        mov             v24.16b,  v21.16b
        b               8f

7:
        // All registers up to v21 are filled already, 3 valid rows left.
        // < 4 valid rows left; fill in padding and filter the last
        // few rows.
        tst             w6,  #8 // LR_HAVE_BOTTOM
        b.eq            71f
        // LR_HAVE_BOTTOM; load 2 rows of edge.
        ld1             {v22.8h}, [x2], x7
        ld1             {v23.8h}, [x2], x7
        mov             v24.16b,  v23.16b
        b               8f
71:
        // !LR_HAVE_BOTTOM, pad 3 rows
        mov             v22.16b,  v21.16b
        mov             v23.16b,  v21.16b
        mov             v24.16b,  v21.16b

8:      // At this point, all registers up to v22-v24 are loaded with
        // edge/padding (depending on how many rows are left).
        filter          0 // This branches to 9f when done
        mov             v22.16b,  v23.16b
        mov             v23.16b,  v24.16b
        b               8b

9:      // End of one vertical slice.
        subs            w3,  w3,  #8
        b.le            0f
        // Move pointers back up to the top and loop horizontally.
        msub            x0,  x1,  x8,  x0
        msub            x2,  x7,  x11, x2
        add             x0,  x0,  #8
        add             x2,  x2,  #16
        mov             w4,  w8
        b               1b

0:
        ret
.purgem filter
endfunc

#define SUM_STRIDE (384+16)

#include "looprestoration_tmpl.S"

// void dav1d_sgr_box3_h_8bpc_neon(int32_t *sumsq, int16_t *sum,
//                                 const pixel (*left)[4],
//                                 const pixel *src, const ptrdiff_t stride,
//                                 const int w, const int h,
//                                 const enum LrEdgeFlags edges);
function sgr_box3_h_8bpc_neon, export=1
        add             w5,  w5,  #2 // w += 2

        // Set up pointers for reading/writing alternate rows
        add             x10, x0,  #(4*SUM_STRIDE)   // sumsq
        add             x11, x1,  #(2*SUM_STRIDE)   // sum
        add             x12, x3,  x4                // src
        lsl             x4,  x4,  #1
        mov             x9,       #(2*2*SUM_STRIDE) // double sum stride

        // Subtract the aligned width from the output stride.
        add             w13, w5,  #7
        bic             w13, w13, #7
        sub             x9,  x9,  w13, uxtw #1

        // Store the width for the vertical loop
        mov             w8,  w5

        // Subtract the number of pixels read from the input from the stride
        add             w13, w13, #8
        sub             x4,  x4,  w13, uxtw

        // Set up the src pointers to include the left edge, for LR_HAVE_LEFT, left == NULL
        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            2f
        // LR_HAVE_LEFT
        cbnz            x2,  0f
        // left == NULL
        sub             x3,  x3,  #2
        sub             x12, x12, #2
        b               1f
0:      // LR_HAVE_LEFT, left != NULL
2:      // !LR_HAVE_LEFT, increase the stride.
        // For this case we don't read the left 2 pixels from the src pointer,
        // but shift it as if we had done that.
        add             x4,  x4,  #2


1:      // Loop vertically
        ld1             {v0.16b},  [x3],  #16
        ld1             {v4.16b},  [x12], #16

        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            0f
        cbz             x2,  2f
        // LR_HAVE_LEFT, left != NULL
        ld1             {v1.s}[3],  [x2], #4
        // Move x3/x12 back to account for the last 2 bytes we loaded earlier,
        // which we'll shift out.
        sub             x3,  x3,  #2
        sub             x12, x12, #2
        ld1             {v5.s}[3],  [x2], #4
        ext             v0.16b, v1.16b, v0.16b, #14
        ext             v4.16b, v5.16b, v4.16b, #14
        b               2f
0:
        // !LR_HAVE_LEFT, fill v1 with the leftmost byte
        // and shift v0 to have 2x the first byte at the front.
        dup             v1.16b, v0.b[0]
        dup             v5.16b, v4.b[0]
        // Move x3 back to account for the last 2 bytes we loaded before,
        // which we shifted out.
        sub             x3,  x3,  #2
        sub             x12, x12, #2
        ext             v0.16b, v1.16b, v0.16b, #14
        ext             v4.16b, v5.16b, v4.16b, #14

2:
        umull           v1.8h,   v0.8b,   v0.8b
        umull2          v2.8h,   v0.16b,  v0.16b
        umull           v5.8h,   v4.8b,   v4.8b
        umull2          v6.8h,   v4.16b,  v4.16b

        tst             w7,  #2 // LR_HAVE_RIGHT
        b.ne            4f
        // If we'll need to pad the right edge, load that byte to pad with
        // here since we can find it pretty easily from here.
        sub             w13, w5, #(2 + 16 - 2 + 1)
        ldr             b30, [x3,  w13, sxtw]
        ldr             b31, [x12, w13, sxtw]
        // Fill v30/v31 with the right padding pixel
        dup             v30.16b, v30.b[0]
        dup             v31.16b, v31.b[0]
3:      // !LR_HAVE_RIGHT

        // Check whether we need to pad the right edge
        cmp             w5,  #10
        b.ge            4f   // If w >= 10, all used input pixels are valid

        // 1 <= w < 10, w pixels valid in v0. For w=9, this ends up called
        // again; it's not strictly needed in those cases (we pad enough here),
        // but keeping the code as simple as possible.

        // Insert padding in v0/4.b[w] onwards
        movrel          x13, right_ext_mask
        sub             x13, x13, w5,  uxtw
        ld1             {v29.16b}, [x13]

        bit             v0.16b,  v30.16b, v29.16b
        bit             v4.16b,  v31.16b, v29.16b

        // Update the precalculated squares
        umull           v1.8h,   v0.8b,   v0.8b
        umull2          v2.8h,   v0.16b,  v0.16b
        umull           v5.8h,   v4.8b,   v4.8b
        umull2          v6.8h,   v4.16b,  v4.16b

4:      // Loop horizontally
        ext             v16.16b, v0.16b,  v0.16b, #1
        ext             v17.16b, v0.16b,  v0.16b, #2
        ext             v18.16b, v4.16b,  v4.16b, #1
        ext             v19.16b, v4.16b,  v4.16b, #2
        uaddl           v3.8h,   v0.8b,   v16.8b
        uaddw           v3.8h,   v3.8h,   v17.8b
        uaddl           v7.8h,   v4.8b,   v18.8b
        uaddw           v7.8h,   v7.8h,   v19.8b

        ext             v20.16b, v1.16b,  v2.16b, #2
        ext             v21.16b, v1.16b,  v2.16b, #4
        ext             v22.16b, v5.16b,  v6.16b, #2
        ext             v23.16b, v5.16b,  v6.16b, #4

        uaddl           v26.4s,  v1.4h,   v20.4h
        uaddl2          v27.4s,  v1.8h,   v20.8h
        uaddw           v26.4s,  v26.4s,  v21.4h
        uaddw2          v27.4s,  v27.4s,  v21.8h

        uaddl           v28.4s,  v5.4h,   v22.4h
        uaddl2          v29.4s,  v5.8h,   v22.8h
        uaddw           v28.4s,  v28.4s,  v23.4h
        uaddw2          v29.4s,  v29.4s,  v23.8h

        subs            w5,  w5,  #8

        st1             {v3.8h},         [x1],  #16
        st1             {v7.8h},         [x11], #16
        st1             {v26.4s,v27.4s}, [x0],  #32
        st1             {v28.4s,v29.4s}, [x10], #32

        b.le            9f
        tst             w7,  #2 // LR_HAVE_RIGHT
        ld1             {v3.8b},  [x3],  #8
        ld1             {v7.8b},  [x12], #8
        mov             v1.16b,  v2.16b
        mov             v5.16b,  v6.16b
        ext             v0.16b,  v0.16b,  v3.16b, #8
        ext             v4.16b,  v4.16b,  v7.16b, #8
        umull           v2.8h,   v3.8b,   v3.8b
        umull           v6.8h,   v7.8b,   v7.8b

        b.ne            4b // If we don't need to pad, just keep summing.
        b               3b // If we need to pad, check how many pixels we have left.

9:
        subs            w6,  w6,  #2
        b.le            0f
        // Jump to the next row and loop horizontally
        add             x0,  x0,  x9, lsl #1
        add             x10, x10, x9, lsl #1
        add             x1,  x1,  x9
        add             x11, x11, x9
        add             x3,  x3,  x4
        add             x12, x12, x4
        mov             w5,  w8
        b               1b
0:
        ret
endfunc

// void dav1d_sgr_box5_h_8bpc_neon(int32_t *sumsq, int16_t *sum,
//                                 const pixel (*left)[4],
//                                 const pixel *src, const ptrdiff_t stride,
//                                 const int w, const int h,
//                                 const enum LrEdgeFlags edges);
function sgr_box5_h_8bpc_neon, export=1
        add             w5,  w5,  #2 // w += 2

        // Set up pointers for reading/writing alternate rows
        add             x10, x0,  #(4*SUM_STRIDE)   // sumsq
        add             x11, x1,  #(2*SUM_STRIDE)   // sum
        add             x12, x3,  x4                // src
        lsl             x4,  x4,  #1
        mov             x9,       #(2*2*SUM_STRIDE) // double sum stride

        // Subtract the aligned width from the output stride.
        add             w13, w5,  #7
        bic             w13, w13, #7
        sub             x9,  x9,  w13, uxtw #1
        add             w13, w13, #8
        sub             x4,  x4,  w13, uxtw

        // Store the width for the vertical loop
        mov             w8,  w5

        // Set up the src pointers to include the left edge, for LR_HAVE_LEFT, left == NULL
        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            2f
        // LR_HAVE_LEFT
        cbnz            x2,  0f
        // left == NULL
        sub             x3,  x3,  #3
        sub             x12, x12, #3
        b               1f
0:      // LR_HAVE_LEFT, left != NULL
2:      // !LR_HAVE_LEFT, increase the stride.
        // For this case we don't read the left 3 pixels from the src pointer,
        // but shift it as if we had done that.
        add             x4,  x4,  #3

1:      // Loop vertically
        ld1             {v0.16b},  [x3],  #16
        ld1             {v4.16b},  [x12], #16

        tst             w7,  #1 // LR_HAVE_LEFT
        b.eq            0f
        cbz             x2,  2f
        // LR_HAVE_LEFT, left != NULL
        ld1             {v1.s}[3],  [x2], #4
        // Move x3/x12 back to account for the last 3 bytes we loaded earlier,
        // which we'll shift out.
        sub             x3,  x3,  #3
        sub             x12, x12, #3
        ld1             {v5.s}[3],  [x2], #4
        ext             v0.16b, v1.16b, v0.16b, #13
        ext             v4.16b, v5.16b, v4.16b, #13
        b               2f
0:
        // !LR_HAVE_LEFT, fill v1 with the leftmost byte
        // and shift v0 to have 3x the first byte at the front.
        dup             v1.16b, v0.b[0]
        dup             v5.16b, v4.b[0]
        // Move x3 back to account for the last 3 bytes we loaded before,
        // which we shifted out.
        sub             x3,  x3,  #3
        sub             x12, x12, #3
        ext             v0.16b, v1.16b, v0.16b, #13
        ext             v4.16b, v5.16b, v4.16b, #13

2:
        umull           v1.8h,   v0.8b,   v0.8b
        umull2          v2.8h,   v0.16b,  v0.16b
        umull           v5.8h,   v4.8b,   v4.8b
        umull2          v6.8h,   v4.16b,  v4.16b

        tst             w7,  #2 // LR_HAVE_RIGHT
        b.ne            4f
        // If we'll need to pad the right edge, load that byte to pad with
        // here since we can find it pretty easily from here.
        sub             w13, w5, #(2 + 16 - 3 + 1)
        ldr             b30, [x3,  w13, sxtw]
        ldr             b31, [x12, w13, sxtw]
        // Fill v30/v31 with the right padding pixel
        dup             v30.16b, v30.b[0]
        dup             v31.16b, v31.b[0]
3:      // !LR_HAVE_RIGHT

        // Check whether we need to pad the right edge
        cmp             w5,  #11
        b.ge            4f   // If w >= 11, all used input pixels are valid

        // 1 <= w < 11, w+1 pixels valid in v0. For w=9 or w=10,
        // this ends up called again; it's not strictly needed in those
        // cases (we pad enough here), but keeping the code as simple as possible.

        // Insert padding in v0/4.b[w+1] onwards; fuse the +1 into the
        // buffer pointer.
        movrel          x13, right_ext_mask, -1
        sub             x13, x13, w5,  uxtw
        ld1             {v29.16b}, [x13]

        bit             v0.16b,  v30.16b, v29.16b
        bit             v4.16b,  v31.16b, v29.16b

        // Update the precalculated squares
        umull           v1.8h,   v0.8b,   v0.8b
        umull2          v2.8h,   v0.16b,  v0.16b
        umull           v5.8h,   v4.8b,   v4.8b
        umull2          v6.8h,   v4.16b,  v4.16b

4:      // Loop horizontally
        ext             v16.16b, v0.16b,  v0.16b, #1
        ext             v17.16b, v0.16b,  v0.16b, #2
        ext             v18.16b, v0.16b,  v0.16b, #3
        ext             v19.16b, v0.16b,  v0.16b, #4
        ext             v20.16b, v4.16b,  v4.16b, #1
        ext             v21.16b, v4.16b,  v4.16b, #2
        ext             v22.16b, v4.16b,  v4.16b, #3
        ext             v23.16b, v4.16b,  v4.16b, #4
        uaddl           v3.8h,   v0.8b,   v16.8b
        uaddl           v24.8h,  v17.8b,  v18.8b
        uaddl           v7.8h,   v4.8b,   v20.8b
        uaddw           v3.8h,   v3.8h,   v19.8b
        uaddl           v25.8h,  v21.8b,  v22.8b
        uaddw           v7.8h,   v7.8h,   v23.8b
        add             v3.8h,   v3.8h,   v24.8h
        add             v7.8h,   v7.8h,   v25.8h

        ext             v16.16b, v1.16b,  v2.16b, #2
        ext             v17.16b, v1.16b,  v2.16b, #4
        ext             v18.16b, v1.16b,  v2.16b, #6
        ext             v19.16b, v1.16b,  v2.16b, #8
        ext             v20.16b, v5.16b,  v6.16b, #2
        ext             v21.16b, v5.16b,  v6.16b, #4
        ext             v22.16b, v5.16b,  v6.16b, #6
        ext             v23.16b, v5.16b,  v6.16b, #8

        uaddl           v26.4s,  v1.4h,   v16.4h
        uaddl2          v27.4s,  v1.8h,   v16.8h
        uaddl           v16.4s,  v17.4h,  v18.4h
        uaddl2          v17.4s,  v17.8h,  v18.8h
        uaddl           v28.4s,  v5.4h,   v20.4h
        uaddl2          v29.4s,  v5.8h,   v20.8h
        uaddw           v26.4s,  v26.4s,  v19.4h
        uaddw2          v27.4s,  v27.4s,  v19.8h
        uaddl           v20.4s,  v21.4h,  v22.4h
        uaddl2          v21.4s,  v21.8h,  v22.8h
        uaddw           v28.4s,  v28.4s,  v23.4h
        uaddw2          v29.4s,  v29.4s,  v23.8h
        add             v26.4s,  v26.4s,  v16.4s
        add             v27.4s,  v27.4s,  v17.4s
        add             v28.4s,  v28.4s,  v20.4s
        add             v29.4s,  v29.4s,  v21.4s

        subs            w5,  w5,  #8

        st1             {v3.8h},         [x1],  #16
        st1             {v7.8h},         [x11], #16
        st1             {v26.4s,v27.4s}, [x0],  #32
        st1             {v28.4s,v29.4s}, [x10], #32

        b.le            9f
        tst             w7,  #2 // LR_HAVE_RIGHT
        ld1             {v3.8b},  [x3],  #8
        ld1             {v7.8b},  [x12], #8
        mov             v1.16b,  v2.16b
        mov             v5.16b,  v6.16b
        ext             v0.16b,  v0.16b,  v3.16b, #8
        ext             v4.16b,  v4.16b,  v7.16b, #8
        umull           v2.8h,   v3.8b,   v3.8b
        umull           v6.8h,   v7.8b,   v7.8b
        b.ne            4b // If we don't need to pad, just keep summing.
        b               3b // If we need to pad, check how many pixels we have left.

9:
        subs            w6,  w6,  #2
        b.le            0f
        // Jump to the next row and loop horizontally
        add             x0,  x0,  x9, lsl #1
        add             x10, x10, x9, lsl #1
        add             x1,  x1,  x9
        add             x11, x11, x9
        add             x3,  x3,  x4
        add             x12, x12, x4
        mov             w5,  w8
        b               1b
0:
        ret
endfunc

sgr_funcs 8
