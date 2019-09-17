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

// void ipred_dc_128_neon(pixel *dst, const ptrdiff_t stride,
//                        const pixel *const topleft,
//                        const int width, const int height, const int a,
//                        const int max_width, const int max_height);
function ipred_dc_128_neon, export=1
        clz             w3,  w3
        adr             x5,  L(ipred_dc_128_tbl)
        sub             w3,  w3,  #25
        ldrh            w3,  [x5, w3, uxtw #1]
        movi            v0.16b,  #128
        sub             x5,  x5,  w3, uxtw
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5
4:
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            4b
        ret
8:
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            8b
        ret
16:
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            16b
        ret
320:
        movi            v1.16b,  #128
32:
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        b.gt            32b
        ret
640:
        movi            v1.16b,  #128
        movi            v2.16b,  #128
        movi            v3.16b,  #128
64:
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        b.gt            64b
        ret

L(ipred_dc_128_tbl):
        .hword L(ipred_dc_128_tbl) - 640b
        .hword L(ipred_dc_128_tbl) - 320b
        .hword L(ipred_dc_128_tbl) -  16b
        .hword L(ipred_dc_128_tbl) -   8b
        .hword L(ipred_dc_128_tbl) -   4b
endfunc

// void ipred_v_neon(pixel *dst, const ptrdiff_t stride,
//                   const pixel *const topleft,
//                   const int width, const int height, const int a,
//                   const int max_width, const int max_height);
function ipred_v_neon, export=1
        clz             w3,  w3
        adr             x5,  L(ipred_v_tbl)
        sub             w3,  w3,  #25
        ldrh            w3,  [x5, w3, uxtw #1]
        add             x2,  x2,  #1
        sub             x5,  x5,  w3, uxtw
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5
40:
        ld1             {v0.s}[0],  [x2]
4:
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            4b
        ret
80:
        ld1             {v0.8b},  [x2]
8:
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            8b
        ret
160:
        ld1             {v0.16b}, [x2], #16
16:
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            16b
        ret
320:
        ld1             {v0.16b, v1.16b}, [x2]
32:
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        b.gt            32b
        ret
640:
        ld1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x2]
64:
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        b.gt            64b
        ret

L(ipred_v_tbl):
        .hword L(ipred_v_tbl) - 640b
        .hword L(ipred_v_tbl) - 320b
        .hword L(ipred_v_tbl) - 160b
        .hword L(ipred_v_tbl) -  80b
        .hword L(ipred_v_tbl) -  40b
endfunc

// void ipred_h_neon(pixel *dst, const ptrdiff_t stride,
//                   const pixel *const topleft,
//                   const int width, const int height, const int a,
//                   const int max_width, const int max_height);
function ipred_h_neon, export=1
        clz             w3,  w3
        adr             x5,  L(ipred_h_tbl)
        sub             w3,  w3,  #25
        ldrh            w3,  [x5, w3, uxtw #1]
        sub             x2,  x2,  #4
        sub             x5,  x5,  w3, uxtw
        mov             x7,  #-4
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5
4:
        ld4r            {v0.8b, v1.8b, v2.8b, v3.8b},  [x2], x7
        st1             {v3.s}[0],  [x0], x1
        st1             {v2.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v1.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            4b
        ret
8:
        ld4r            {v0.8b, v1.8b, v2.8b, v3.8b},  [x2], x7
        st1             {v3.8b},  [x0], x1
        st1             {v2.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v1.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            8b
        ret
16:
        ld4r            {v0.16b, v1.16b, v2.16b, v3.16b},  [x2], x7
        st1             {v3.16b}, [x0], x1
        st1             {v2.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v1.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            16b
        ret
32:
        ld4r            {v0.16b, v1.16b, v2.16b, v3.16b},  [x2], x7
        str             q3,  [x0, #16]
        str             q2,  [x6, #16]
        st1             {v3.16b}, [x0], x1
        st1             {v2.16b}, [x6], x1
        subs            w4,  w4,  #4
        str             q1,  [x0, #16]
        str             q0,  [x6, #16]
        st1             {v1.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            32b
        ret
64:
        ld4r            {v0.16b, v1.16b, v2.16b, v3.16b},  [x2], x7
        str             q3,  [x0, #16]
        str             q2,  [x6, #16]
        stp             q3,  q3,  [x0, #32]
        stp             q2,  q2,  [x6, #32]
        st1             {v3.16b}, [x0], x1
        st1             {v2.16b}, [x6], x1
        subs            w4,  w4,  #4
        str             q1,  [x0, #16]
        str             q0,  [x6, #16]
        stp             q1,  q1,  [x0, #32]
        stp             q0,  q0,  [x6, #32]
        st1             {v1.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            64b
        ret

L(ipred_h_tbl):
        .hword L(ipred_h_tbl) - 64b
        .hword L(ipred_h_tbl) - 32b
        .hword L(ipred_h_tbl) - 16b
        .hword L(ipred_h_tbl) -  8b
        .hword L(ipred_h_tbl) -  4b
endfunc

// void ipred_dc_top_neon(pixel *dst, const ptrdiff_t stride,
//                        const pixel *const topleft,
//                        const int width, const int height, const int a,
//                        const int max_width, const int max_height);
function ipred_dc_top_neon, export=1
        clz             w3,  w3
        adr             x5,  L(ipred_dc_top_tbl)
        sub             w3,  w3,  #25
        ldrh            w3,  [x5, w3, uxtw #1]
        add             x2,  x2,  #1
        sub             x5,  x5,  w3, uxtw
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5
40:
        ld1r            {v0.2s},  [x2]
        uaddlv          h0,      v0.8b
        rshrn           v0.8b,   v0.8h,   #3
        dup             v0.8b,   v0.b[0]
4:
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            4b
        ret
80:
        ld1             {v0.8b},  [x2]
        uaddlv          h0,      v0.8b
        rshrn           v0.8b,   v0.8h,   #3
        dup             v0.8b,   v0.b[0]
8:
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            8b
        ret
160:
        ld1             {v0.16b}, [x2]
        uaddlv          h0,      v0.16b
        rshrn           v0.8b,   v0.8h,   #4
        dup             v0.16b,  v0.b[0]
16:
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            16b
        ret
320:
        ld1             {v0.16b, v1.16b}, [x2]
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        add             v2.4h,   v0.4h,   v1.4h
        rshrn           v2.8b,   v2.8h,   #5
        dup             v0.16b,  v2.b[0]
        dup             v1.16b,  v2.b[0]
32:
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        b.gt            32b
        ret
640:
        ld1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x2]
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        uaddlv          h2,      v2.16b
        uaddlv          h3,      v3.16b
        add             v4.4h,   v0.4h,   v1.4h
        add             v5.4h,   v2.4h,   v3.4h
        add             v4.4h,   v4.4h,   v5.4h
        rshrn           v4.8b,   v4.8h,   #6
        dup             v0.16b,  v4.b[0]
        dup             v1.16b,  v4.b[0]
        dup             v2.16b,  v4.b[0]
        dup             v3.16b,  v4.b[0]
64:
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        b.gt            64b
        ret

L(ipred_dc_top_tbl):
        .hword L(ipred_dc_top_tbl) - 640b
        .hword L(ipred_dc_top_tbl) - 320b
        .hword L(ipred_dc_top_tbl) - 160b
        .hword L(ipred_dc_top_tbl) -  80b
        .hword L(ipred_dc_top_tbl) -  40b
endfunc

// void ipred_dc_left_neon(pixel *dst, const ptrdiff_t stride,
//                         const pixel *const topleft,
//                         const int width, const int height, const int a,
//                         const int max_width, const int max_height);
function ipred_dc_left_neon, export=1
        sub             x2,  x2,  w4, uxtw
        clz             w3,  w3
        clz             w7,  w4
        adr             x5,  L(ipred_dc_left_tbl)
        sub             w3,  w3,  #20 // 25 leading bits, minus table offset 5
        sub             w7,  w7,  #25
        ldrh            w3,  [x5, w3, uxtw #1]
        ldrh            w7,  [x5, w7, uxtw #1]
        sub             x3,  x5,  w3, uxtw
        sub             x5,  x5,  w7, uxtw
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5

L(ipred_dc_left_h4):
        ld1r            {v0.2s},  [x2]
        uaddlv          h0,      v0.8b
        rshrn           v0.8b,   v0.8h,   #3
        dup             v0.16b,  v0.b[0]
        br              x3
L(ipred_dc_left_w4):
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            L(ipred_dc_left_w4)
        ret

L(ipred_dc_left_h8):
        ld1             {v0.8b},  [x2]
        uaddlv          h0,      v0.8b
        rshrn           v0.8b,   v0.8h,   #3
        dup             v0.16b,  v0.b[0]
        br              x3
L(ipred_dc_left_w8):
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            L(ipred_dc_left_w8)
        ret

L(ipred_dc_left_h16):
        ld1             {v0.16b}, [x2]
        uaddlv          h0,      v0.16b
        rshrn           v0.8b,   v0.8h,   #4
        dup             v0.16b,  v0.b[0]
        br              x3
L(ipred_dc_left_w16):
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            L(ipred_dc_left_w16)
        ret

L(ipred_dc_left_h32):
        ld1             {v0.16b, v1.16b}, [x2]
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        add             v0.4h,   v0.4h,   v1.4h
        rshrn           v0.8b,   v0.8h,   #5
        dup             v0.16b,  v0.b[0]
        br              x3
L(ipred_dc_left_w32):
        mov             v1.16b,  v0.16b
1:
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        b.gt            1b
        ret

L(ipred_dc_left_h64):
        ld1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x2]
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        uaddlv          h2,      v2.16b
        uaddlv          h3,      v3.16b
        add             v0.4h,   v0.4h,   v1.4h
        add             v2.4h,   v2.4h,   v3.4h
        add             v0.4h,   v0.4h,   v2.4h
        rshrn           v0.8b,   v0.8h,   #6
        dup             v0.16b,  v0.b[0]
        br              x3
L(ipred_dc_left_w64):
        mov             v1.16b,  v0.16b
        mov             v2.16b,  v0.16b
        mov             v3.16b,  v0.16b
1:
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        b.gt            1b
        ret

L(ipred_dc_left_tbl):
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_h64)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_h32)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_h16)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_h8)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_h4)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_w64)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_w32)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_w16)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_w8)
        .hword L(ipred_dc_left_tbl) - L(ipred_dc_left_w4)
endfunc

// void ipred_dc_neon(pixel *dst, const ptrdiff_t stride,
//                    const pixel *const topleft,
//                    const int width, const int height, const int a,
//                    const int max_width, const int max_height);
function ipred_dc_neon, export=1
        sub             x2,  x2,  w4, uxtw
        add             w7,  w3,  w4             // width + height
        clz             w3,  w3
        clz             w6,  w4
        dup             v16.8h, w7               // width + height
        adr             x5,  L(ipred_dc_tbl)
        rbit            w7,  w7                  // rbit(width + height)
        sub             w3,  w3,  #20            // 25 leading bits, minus table offset 5
        sub             w6,  w6,  #25
        clz             w7,  w7                  // ctz(width + height)
        ldrh            w3,  [x5, w3, uxtw #1]
        ldrh            w6,  [x5, w6, uxtw #1]
        neg             w7,  w7                  // -ctz(width + height)
        sub             x3,  x5,  w3, uxtw
        sub             x5,  x5,  w6, uxtw
        ushr            v16.8h,  v16.8h,  #1     // (width + height) >> 1
        dup             v17.8h,  w7              // -ctz(width + height)
        add             x6,  x0,  x1
        lsl             x1,  x1,  #1
        br              x5

L(ipred_dc_h4):
        ld1             {v0.s}[0],  [x2], #4
        ins             v0.s[1], wzr
        uaddlv          h0,      v0.8b
        br              x3
L(ipred_dc_w4):
        add             x2,  x2,  #1
        ld1             {v1.s}[0],  [x2]
        ins             v1.s[1], wzr
        add             v0.4h,   v0.4h,   v16.4h
        uaddlv          h1,      v1.8b
        cmp             w4,  #4
        add             v0.4h,   v0.4h,   v1.4h
        ushl            v0.4h,   v0.4h,   v17.4h
        b.eq            1f
        // h = 8/16
        mov             w16, #(0x3334/2)
        movk            w16, #(0x5556/2), lsl #16
        add             w17, w4,  w4  // w17 = 2*h = 16 or 32
        lsr             w16, w16, w17
        dup             v16.4h,  w16
        sqdmulh         v0.4h,   v0.4h,   v16.4h
1:
        dup             v0.8b,   v0.b[0]
2:
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.s}[0],  [x0], x1
        st1             {v0.s}[0],  [x6], x1
        b.gt            2b
        ret

L(ipred_dc_h8):
        ld1             {v0.8b},  [x2], #8
        uaddlv          h0,      v0.8b
        br              x3
L(ipred_dc_w8):
        add             x2,  x2,  #1
        ld1             {v1.8b},  [x2]
        add             v0.4h,   v0.4h,   v16.4h
        uaddlv          h1,      v1.8b
        cmp             w4,  #8
        add             v0.4h,   v0.4h,   v1.4h
        ushl            v0.4h,   v0.4h,   v17.4h
        b.eq            1f
        // h = 4/16/32
        cmp             w4,  #32
        mov             w16, #(0x3334/2)
        mov             w17, #(0x5556/2)
        csel            w16, w16, w17, eq
        dup             v16.4h,  w16
        sqdmulh         v0.4h,   v0.4h,   v16.4h
1:
        dup             v0.8b,   v0.b[0]
2:
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.8b},  [x0], x1
        st1             {v0.8b},  [x6], x1
        b.gt            2b
        ret

L(ipred_dc_h16):
        ld1             {v0.16b}, [x2], #16
        uaddlv          h0,      v0.16b
        br              x3
L(ipred_dc_w16):
        add             x2,  x2,  #1
        ld1             {v1.16b}, [x2]
        add             v0.4h,   v0.4h,   v16.4h
        uaddlv          h1,      v1.16b
        cmp             w4,  #16
        add             v0.4h,   v0.4h,   v1.4h
        ushl            v0.4h,   v0.4h,   v17.4h
        b.eq            1f
        // h = 4/8/32/64
        tst             w4,  #(32+16+8) // 16 added to make a consecutive bitmask
        mov             w16, #(0x3334/2)
        mov             w17, #(0x5556/2)
        csel            w16, w16, w17, eq
        dup             v16.4h,  w16
        sqdmulh         v0.4h,   v0.4h,   v16.4h
1:
        dup             v0.16b,  v0.b[0]
2:
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b}, [x0], x1
        st1             {v0.16b}, [x6], x1
        b.gt            2b
        ret

L(ipred_dc_h32):
        ld1             {v0.16b, v1.16b}, [x2], #32
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        add             v0.4h,   v0.4h,   v1.4h
        br              x3
L(ipred_dc_w32):
        add             x2,  x2,  #1
        ld1             {v1.16b, v2.16b}, [x2]
        add             v0.4h,   v0.4h,   v16.4h
        uaddlv          h1,      v1.16b
        uaddlv          h2,      v2.16b
        cmp             w4,  #32
        add             v0.4h,   v0.4h,   v1.4h
        add             v0.4h,   v0.4h,   v2.4h
        ushl            v0.4h,   v0.4h,   v17.4h
        b.eq            1f
        // h = 8/16/64
        cmp             w4,  #8
        mov             w16, #(0x3334/2)
        mov             w17, #(0x5556/2)
        csel            w16, w16, w17, eq
        dup             v16.4h,  w16
        sqdmulh         v0.4h,   v0.4h,   v16.4h
1:
        dup             v0.16b,  v0.b[0]
        dup             v1.16b,  v0.b[0]
2:
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b}, [x0], x1
        st1             {v0.16b, v1.16b}, [x6], x1
        b.gt            2b
        ret

L(ipred_dc_h64):
        ld1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64
        uaddlv          h0,      v0.16b
        uaddlv          h1,      v1.16b
        uaddlv          h2,      v2.16b
        uaddlv          h3,      v3.16b
        add             v0.4h,   v0.4h,   v1.4h
        add             v2.4h,   v2.4h,   v3.4h
        add             v0.4h,   v0.4h,   v2.4h
        br              x3
L(ipred_dc_w64):
        mov             v1.16b,  v0.16b
        mov             v2.16b,  v0.16b
        mov             v3.16b,  v0.16b
2:
        add             x2,  x2,  #1
        ld1             {v1.16b, v2.16b, v3.16b, v4.16b}, [x2]
        add             v0.4h,   v0.4h,   v16.4h
        uaddlv          h1,      v1.16b
        uaddlv          h2,      v2.16b
        uaddlv          h3,      v3.16b
        uaddlv          h4,      v4.16b
        add             v1.4h,   v1.4h,   v2.4h
        add             v3.4h,   v3.4h,   v4.4h
        cmp             w4,  #64
        add             v0.4h,   v0.4h,   v1.4h
        add             v0.4h,   v0.4h,   v3.4h
        ushl            v0.4h,   v0.4h,   v17.4h
        b.eq            1f
        // h = 16/32
        mov             w16, #(0x5556/2)
        movk            w16, #(0x3334/2), lsl #16
        lsr             w16, w16, w4
        dup             v16.4h,  w16
        sqdmulh         v0.4h,   v0.4h,   v16.4h
1:
        dup             v0.16b,  v0.b[0]
        dup             v1.16b,  v0.b[0]
        dup             v2.16b,  v0.b[0]
        dup             v3.16b,  v0.b[0]
2:
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        subs            w4,  w4,  #4
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], x1
        st1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x6], x1
        b.gt            2b
        ret

L(ipred_dc_tbl):
        .hword L(ipred_dc_tbl) - L(ipred_dc_h64)
        .hword L(ipred_dc_tbl) - L(ipred_dc_h32)
        .hword L(ipred_dc_tbl) - L(ipred_dc_h16)
        .hword L(ipred_dc_tbl) - L(ipred_dc_h8)
        .hword L(ipred_dc_tbl) - L(ipred_dc_h4)
        .hword L(ipred_dc_tbl) - L(ipred_dc_w64)
        .hword L(ipred_dc_tbl) - L(ipred_dc_w32)
        .hword L(ipred_dc_tbl) - L(ipred_dc_w16)
        .hword L(ipred_dc_tbl) - L(ipred_dc_w8)
        .hword L(ipred_dc_tbl) - L(ipred_dc_w4)
endfunc
