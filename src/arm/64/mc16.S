/*
 * Copyright © 2018, VideoLAN and dav1d authors
 * Copyright © 2018, Janne Grunau
 * Copyright © 2020, Martin Storsjo
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

#define PREP_BIAS 8192

.macro avg d0, d1, t0, t1, t2, t3
        ld1             {\t0\().8h,\t1\().8h},  [x2],  32
        ld1             {\t2\().8h,\t3\().8h},  [x3],  32
        sqadd           \t0\().8h,  \t0\().8h,  \t2\().8h
        sqadd           \t1\().8h,  \t1\().8h,  \t3\().8h
        smax            \t0\().8h,  \t0\().8h,  v28.8h // -2*PREP_BIAS - 1 << intermediate_bits
        smax            \t1\().8h,  \t1\().8h,  v28.8h // -2*PREP_BIAS - 1 << intermediate_bits
        sqsub           \t0\().8h,  \t0\().8h,  v28.8h // -2*PREP_BIAS - 1 << intermediate_bits
        sqsub           \t1\().8h,  \t1\().8h,  v28.8h // -2*PREP_BIAS - 1 << intermediate_bits
        sshl            \d0\().8h,  \t0\().8h,  v29.8h // -(intermediate_bits+1)
        sshl            \d1\().8h,  \t1\().8h,  v29.8h // -(intermediate_bits+1)
.endm

.macro w_avg d0, d1, t0, t1, t2, t3
        ld1             {\t0\().8h,\t1\().8h},  [x2],  32
        ld1             {\t2\().8h,\t3\().8h},  [x3],  32
        // This difference requires a 17 bit range, and all bits are
        // significant for the following multiplication.
        ssubl           \d0\().4s,  \t2\().4h,  \t0\().4h
        ssubl2          \t0\().4s,  \t2\().8h,  \t0\().8h
        ssubl           \d1\().4s,  \t3\().4h,  \t1\().4h
        ssubl2          \t1\().4s,  \t3\().8h,  \t1\().8h
        mul             \d0\().4s,  \d0\().4s,  v27.4s
        mul             \t0\().4s,  \t0\().4s,  v27.4s
        mul             \d1\().4s,  \d1\().4s,  v27.4s
        mul             \t1\().4s,  \t1\().4s,  v27.4s
        sshr            \d0\().4s,  \d0\().4s,  #4
        sshr            \t0\().4s,  \t0\().4s,  #4
        sshr            \d1\().4s,  \d1\().4s,  #4
        sshr            \t1\().4s,  \t1\().4s,  #4
        saddw           \d0\().4s,  \d0\().4s,  \t2\().4h
        saddw2          \t0\().4s,  \t0\().4s,  \t2\().8h
        saddw           \d1\().4s,  \d1\().4s,  \t3\().4h
        saddw2          \t1\().4s,  \t1\().4s,  \t3\().8h
        xtn             \d0\().4h,  \d0\().4s
        xtn2            \d0\().8h,  \t0\().4s
        xtn             \d1\().4h,  \d1\().4s
        xtn2            \d1\().8h,  \t1\().4s
        srshl           \d0\().8h,  \d0\().8h,  v29.8h // -intermediate_bits
        srshl           \d1\().8h,  \d1\().8h,  v29.8h // -intermediate_bits
        add             \d0\().8h,  \d0\().8h,  v28.8h // PREP_BIAS >> intermediate_bits
        add             \d1\().8h,  \d1\().8h,  v28.8h // PREP_BIAS >> intermediate_bits
        smin            \d0\().8h,  \d0\().8h,  v31.8h // bitdepth_max
        smin            \d1\().8h,  \d1\().8h,  v31.8h // bitdepth_max
        smax            \d0\().8h,  \d0\().8h,  v30.8h // 0
        smax            \d1\().8h,  \d1\().8h,  v30.8h // 0
.endm

.macro mask d0, d1, t0, t1, t2, t3
        ld1             {v27.16b}, [x6],  16
        ld1             {\t0\().8h,\t1\().8h},  [x2],  32
        neg             v27.16b, v27.16b
        ld1             {\t2\().8h,\t3\().8h},  [x3],  32
        sxtl            v26.8h,  v27.8b
        sxtl2           v27.8h,  v27.16b
        sxtl            v24.4s,  v26.4h
        sxtl2           v25.4s,  v26.8h
        sxtl            v26.4s,  v27.4h
        sxtl2           v27.4s,  v27.8h
        ssubl           \d0\().4s,  \t2\().4h,  \t0\().4h
        ssubl2          \t0\().4s,  \t2\().8h,  \t0\().8h
        ssubl           \d1\().4s,  \t3\().4h,  \t1\().4h
        ssubl2          \t1\().4s,  \t3\().8h,  \t1\().8h
        mul             \d0\().4s,  \d0\().4s,  v24.4s
        mul             \t0\().4s,  \t0\().4s,  v25.4s
        mul             \d1\().4s,  \d1\().4s,  v26.4s
        mul             \t1\().4s,  \t1\().4s,  v27.4s
        sshr            \d0\().4s,  \d0\().4s,  #6
        sshr            \t0\().4s,  \t0\().4s,  #6
        sshr            \d1\().4s,  \d1\().4s,  #6
        sshr            \t1\().4s,  \t1\().4s,  #6
        saddw           \d0\().4s,  \d0\().4s,  \t2\().4h
        saddw2          \t0\().4s,  \t0\().4s,  \t2\().8h
        saddw           \d1\().4s,  \d1\().4s,  \t3\().4h
        saddw2          \t1\().4s,  \t1\().4s,  \t3\().8h
        xtn             \d0\().4h,  \d0\().4s
        xtn2            \d0\().8h,  \t0\().4s
        xtn             \d1\().4h,  \d1\().4s
        xtn2            \d1\().8h,  \t1\().4s
        srshl           \d0\().8h,  \d0\().8h,  v29.8h // -intermediate_bits
        srshl           \d1\().8h,  \d1\().8h,  v29.8h // -intermediate_bits
        add             \d0\().8h,  \d0\().8h,  v28.8h // PREP_BIAS >> intermediate_bits
        add             \d1\().8h,  \d1\().8h,  v28.8h // PREP_BIAS >> intermediate_bits
        smin            \d0\().8h,  \d0\().8h,  v31.8h // bitdepth_max
        smin            \d1\().8h,  \d1\().8h,  v31.8h // bitdepth_max
        smax            \d0\().8h,  \d0\().8h,  v30.8h // 0
        smax            \d1\().8h,  \d1\().8h,  v30.8h // 0
.endm

.macro bidir_fn type, bdmax
function \type\()_16bpc_neon, export=1
        clz             w4,  w4
.ifnc \type, avg
        dup             v31.8h,  \bdmax // bitdepth_max
        movi            v30.8h,  #0
.endif
        clz             w7,  \bdmax
        sub             w7,  w7,  #18   // intermediate_bits = clz(bitdepth_max) - 18
.ifc \type, avg
        mov             w9,  #1
        mov             w8,  #-2*PREP_BIAS
        lsl             w9,  w9,  w7    // 1 << intermediate_bits
        add             w7,  w7,  #1
        sub             w8,  w8,  w9    // -2*PREP_BIAS - 1 << intermediate_bits
        neg             w7,  w7         // -(intermediate_bits+1)
        dup             v28.8h,   w8    // -2*PREP_BIAS - 1 << intermediate_bits
        dup             v29.8h,   w7    // -(intermediate_bits+1)
.else
        mov             w8,  #PREP_BIAS
        lsr             w8,  w8,  w7    // PREP_BIAS >> intermediate_bits
        neg             w7,  w7         // -intermediate_bits
        dup             v28.8h,  w8     // PREP_BIAS >> intermediate_bits
        dup             v29.8h,  w7     // -intermediate_bits
.endif
.ifc \type, w_avg
        dup             v27.4s,  w6
        neg             v27.4s,  v27.4s
.endif
        adr             x7,  L(\type\()_tbl)
        sub             w4,  w4,  #24
        \type           v4,  v5,  v0,  v1,  v2,  v3
        ldrh            w4,  [x7, x4, lsl #1]
        sub             x7,  x7,  w4, uxtw
        br              x7
40:
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
4:
        subs            w5,  w5,  #4
        st1             {v4.d}[0],  [x0], x1
        st1             {v4.d}[1],  [x7], x1
        st1             {v5.d}[0],  [x0], x1
        st1             {v5.d}[1],  [x7], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               4b
80:
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
8:
        st1             {v4.8h},  [x0], x1
        subs            w5,  w5,  #2
        st1             {v5.8h},  [x7], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               8b
16:
        \type           v6,  v7,  v0,  v1,  v2,  v3
        st1             {v4.8h, v5.8h}, [x0], x1
        subs            w5,  w5,  #2
        st1             {v6.8h, v7.8h}, [x0], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               16b
32:
        \type           v6,  v7,  v0,  v1,  v2,  v3
        subs            w5,  w5,  #1
        st1             {v4.8h, v5.8h, v6.8h, v7.8h},  [x0], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               32b
640:
        add             x7,  x0,  #64
64:
        \type           v6,  v7,  v0,  v1,  v2,  v3
        \type           v16, v17, v0,  v1,  v2,  v3
        st1             {v4.8h, v5.8h, v6.8h, v7.8h},  [x0], x1
        \type           v18, v19, v0,  v1,  v2,  v3
        subs            w5,  w5,  #1
        st1             {v16.8h,v17.8h,v18.8h,v19.8h}, [x7], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               64b
1280:
        add             x7,  x0,  #64
        mov             x8,  #128
        sub             x1,  x1,  #128
128:
        \type           v6,  v7,  v0,  v1,  v2,  v3
        \type           v16, v17, v0,  v1,  v2,  v3
        st1             {v4.8h, v5.8h, v6.8h, v7.8h},  [x0], x8
        \type           v18, v19, v0,  v1,  v2,  v3
        st1             {v16.8h,v17.8h,v18.8h,v19.8h}, [x7], x8
        \type           v4,  v5,  v0,  v1,  v2,  v3
        \type           v6,  v7,  v0,  v1,  v2,  v3
        \type           v16, v17, v0,  v1,  v2,  v3
        subs            w5,  w5,  #1
        st1             {v4.8h, v5.8h, v6.8h, v7.8h},  [x0], x1
        \type           v18, v19, v0,  v1,  v2,  v3
        st1             {v16.8h,v17.8h,v18.8h,v19.8h}, [x7], x1
        b.le            0f
        \type           v4,  v5,  v0,  v1,  v2,  v3
        b               128b
0:
        ret
L(\type\()_tbl):
        .hword L(\type\()_tbl) - 1280b
        .hword L(\type\()_tbl) -  640b
        .hword L(\type\()_tbl) -   32b
        .hword L(\type\()_tbl) -   16b
        .hword L(\type\()_tbl) -   80b
        .hword L(\type\()_tbl) -   40b
endfunc
.endm

bidir_fn avg, w6
bidir_fn w_avg, w7
bidir_fn mask, w7
