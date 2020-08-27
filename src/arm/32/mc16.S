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

.macro avg d0, d00, d01, d1, d10, d11
        vld1.16         {q0, q1}, [r2, :128]!
        vld1.16         {q2, q3}, [r3, :128]!
        vqadd.s16       q0,  q0,  q2
        vqadd.s16       q1,  q1,  q3
        vmax.s16        q0,  q0,  q12 // -2*PREP_BIAS - 1 << intermediate_bits
        vmax.s16        q1,  q1,  q12 // -2*PREP_BIAS - 1 << intermediate_bits
        vqsub.s16       q0,  q0,  q12 // -2*PREP_BIAS - 1 << intermediate_bits
        vqsub.s16       q1,  q1,  q12 // -2*PREP_BIAS - 1 << intermediate_bits
        vshl.s16        \d0, q0,  q13 // -(intermediate_bits+1)
        vshl.s16        \d1, q1,  q13 // -(intermediate_bits+1)
.endm

.macro w_avg d0, d00, d01, d1, d10, d11
        vld1.16         {q0, q1}, [r2, :128]!
        vld1.16         {q2, q3}, [r3, :128]!
        // This difference requires a 17 bit range, and all bits are
        // significant for the following multiplication.
        vsubl.s16       \d0, d4,  d0
        vsubl.s16       q0,  d5,  d1
        vsubl.s16       \d1, d6,  d2
        vsubl.s16       q1,  d7,  d3
        vmul.s32        \d0, \d0, q4
        vmul.s32        q0,  q0,  q4
        vmul.s32        \d1, \d1, q4
        vmul.s32        q1,  q1,  q4
        vshr.s32        \d0, \d0, #4
        vshr.s32        q0,  q0,  #4
        vshr.s32        \d1, \d1, #4
        vshr.s32        q1,  q1,  #4
        vaddw.s16       \d0, \d0, d4
        vaddw.s16       q0,  q0,  d5
        vaddw.s16       \d1, \d1, d6
        vaddw.s16       q1,  q1,  d7
        vmovn.i32       \d00, \d0
        vmovn.i32       \d01, q0
        vmovn.i32       \d10, \d1
        vmovn.i32       \d11, q1
        vrshl.s16       \d0, \d0, q13 // -intermediate_bits
        vrshl.s16       \d1, \d1, q13 // -intermediate_bits
        vadd.s16        \d0, \d0, q12 // PREP_BIAS >> intermediate_bits
        vadd.s16        \d1, \d1, q12 // PREP_BIAS >> intermediate_bits
        vmin.s16        \d0, \d0, q15 // bitdepth_max
        vmin.s16        \d1, \d1, q15 // bitdepth_max
        vmax.s16        \d0, \d0, q14 // 0
        vmax.s16        \d1, \d1, q14 // 0
.endm

.macro mask d0, d00, d01, d1, d10, d11
        vld1.8          {q7},     [r6, :128]!
        vld1.16         {q0, q1}, [r2, :128]!
        vneg.s8         q7,  q7
        vld1.16         {q2, q3}, [r3, :128]!
        vmovl.s8        q6,  d14
        vmovl.s8        q7,  d15
        vmovl.s16       q4,  d12
        vmovl.s16       q5,  d13
        vmovl.s16       q6,  d14
        vmovl.s16       q7,  d15
        vsubl.s16       \d0, d4,  d0
        vsubl.s16       q0,  d5,  d1
        vsubl.s16       \d1, d6,  d2
        vsubl.s16       q1,  d7,  d3
        vmul.s32        \d0, \d0, q4
        vmul.s32        q0,  q0,  q5
        vmul.s32        \d1, \d1, q6
        vmul.s32        q1,  q1,  q7
        vshr.s32        \d0, \d0, #6
        vshr.s32        q0,  q0,  #6
        vshr.s32        \d1, \d1, #6
        vshr.s32        q1,  q1,  #6
        vaddw.s16       \d0, \d0, d4
        vaddw.s16       q0,  q0,  d5
        vaddw.s16       \d1, \d1, d6
        vaddw.s16       q1,  q1,  d7
        vmovn.i32       \d00, \d0
        vmovn.i32       \d01, q0
        vmovn.i32       \d10, \d1
        vmovn.i32       \d11, q1
        vrshl.s16       \d0, \d0, q13 // -intermediate_bits
        vrshl.s16       \d1, \d1, q13 // -intermediate_bits
        vadd.s16        \d0, \d0, q12 // PREP_BIAS >> intermediate_bits
        vadd.s16        \d1, \d1, q12 // PREP_BIAS >> intermediate_bits
        vmin.s16        \d0, \d0, q15 // bitdepth_max
        vmin.s16        \d1, \d1, q15 // bitdepth_max
        vmax.s16        \d0, \d0, q14 // 0
        vmax.s16        \d1, \d1, q14 // 0
.endm

.macro bidir_fn type, bdmax
function \type\()_16bpc_neon, export=1
        push            {r4-r7,lr}
        ldr             r4,  [sp, #20]
        ldr             r5,  [sp, #24]
        ldr             r6,  [sp, #28]
        clz             r4,  r4
.ifnc \type, avg
        ldr             r7,  [sp, #32]
        vmov.i16        q14, #0
        vdup.16         q15, r7         // bitdepth_max
.endif
.ifc \type, w_avg
        vpush           {q4}
.endif
.ifc \type, mask
        vpush           {q4-q7}
.endif
        clz             r7,  \bdmax
        sub             r7,  r7,  #18   // intermediate_bits = clz(bitdepth_max) - 18
.ifc \type, avg
        mov             lr,  #1
        movw            r12, #2*PREP_BIAS
        lsl             lr,  lr,  r7    // 1 << intermediate_bits
        neg             r12, r12         // -2*PREP_BIAS
        add             r7,  r7,  #1
        sub             r12, r12, lr    // -2*PREP_BIAS - 1 << intermediate_bits
        neg             r7,  r7         // -(intermediate_bits+1)
        vdup.16         q12, r12         // -2*PREP_BIAS - 1 << intermediate_bits
        vdup.16         q13, r7         // -(intermediate_bits+1)
.else
        mov             r12, #PREP_BIAS
        lsr             r12, r12, r7    // PREP_BIAS >> intermediate_bits
        neg             r7,  r7         // -intermediate_bits
        vdup.16         q12, r12         // PREP_BIAS >> intermediate_bits
        vdup.16         q13, r7         // -intermediate_bits
.endif
.ifc \type, w_avg
        vdup.32         q4,  r6
        vneg.s32        q4,  q4
.endif
        adr             r7,  L(\type\()_tbl)
        sub             r4,  r4,  #24
        \type           q8,  d16, d17, q9,  d18, d19
        ldr             r4,  [r7, r4, lsl #2]
        add             r7,  r7,  r4
        bx              r7

        .align 2
L(\type\()_tbl):
        .word 1280f - L(\type\()_tbl) + CONFIG_THUMB
        .word 640f  - L(\type\()_tbl) + CONFIG_THUMB
        .word 320f  - L(\type\()_tbl) + CONFIG_THUMB
        .word 160f  - L(\type\()_tbl) + CONFIG_THUMB
        .word 80f   - L(\type\()_tbl) + CONFIG_THUMB
        .word 40f   - L(\type\()_tbl) + CONFIG_THUMB

40:
        add             r7,  r0,  r1
        lsl             r1,  r1,  #1
4:
        subs            r5,  r5,  #4
        vst1.16         {d16},  [r0, :64], r1
        vst1.16         {d17},  [r7, :64], r1
        vst1.16         {d18},  [r0, :64], r1
        vst1.16         {d19},  [r7, :64], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               4b
80:
        add             r7,  r0,  r1
        lsl             r1,  r1,  #1
8:
        vst1.16         {q8},  [r0, :128], r1
        subs            r5,  r5,  #2
        vst1.16         {q9},  [r7, :128], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               8b
160:
16:
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r1
        subs            r5,  r5,  #2
        vst1.16         {q10, q11}, [r0, :128], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               16b
320:
        add             r7,  r0,  #32
32:
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r1
        subs            r5,  r5,  #1
        vst1.16         {q10, q11}, [r7, :128], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               32b
640:
        add             r7,  r0,  #32
        mov             r12, #64
        sub             r1,  r1,  #64
64:
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r12
        \type           q8,  d16, d17, q9,  d18, d19
        vst1.16         {q10, q11}, [r7, :128], r12
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r1
        subs            r5,  r5,  #1
        vst1.16         {q10, q11}, [r7, :128], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               64b
1280:
        add             r7,  r0,  #32
        mov             r12, #64
        sub             r1,  r1,  #192
128:
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r12
        \type           q8,  d16, d17, q9,  d18, d19
        vst1.16         {q10, q11}, [r7, :128], r12
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r12
        \type           q8,  d16, d17, q9,  d18, d19
        vst1.16         {q10, q11}, [r7, :128], r12
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r12
        \type           q8,  d16, d17, q9,  d18, d19
        vst1.16         {q10, q11}, [r7, :128], r12
        \type           q10, d20, d21, q11, d22, d23
        vst1.16         {q8,  q9},  [r0, :128], r1
        subs            r5,  r5,  #1
        vst1.16         {q10, q11}, [r7, :128], r1
        ble             0f
        \type           q8,  d16, d17, q9,  d18, d19
        b               128b
0:
.ifc \type, mask
        vpop            {q4-q7}
.endif
.ifc \type, w_avg
        vpop            {q4}
.endif
        pop             {r4-r7,pc}
endfunc
.endm

bidir_fn avg, r6
bidir_fn w_avg, r7
bidir_fn mask, r7
