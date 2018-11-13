; Copyright (c) 2018, The rav1e contributors. All rights reserved
;
; This source code is subject to the terms of the BSD 2 Clause License and
; the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
; was not distributed with this source code in the LICENSE file, you can
; obtain it at www.aomedia.org/license/software. If the Alliance for Open
; Media Patent License 1.0 was not distributed with this source code in the
; PATENTS file, you can obtain it at www.aomedia.org/license/patent.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

%macro W_ABS_DIFF 8
    psubw               %1, %5
    psubw               %2, %6
    psubw               %3, %7
    psubw               %4, %8
    pabsw               %1, %1
    pabsw               %2, %2
    pabsw               %3, %3
    pabsw               %4, %4
%endmacro

INIT_XMM ssse3
cglobal sad_4x4_hbd, 4, 6, 8, src, src_stride, dst, dst_stride, \
                              src_stride3, dst_stride3
    lea       src_stride3q, [src_strideq*3]
    lea       dst_stride3q, [dst_strideq*3]
    movq                m0, [srcq]
    movq                m1, [srcq+src_strideq*1]
    movq                m2, [srcq+src_strideq*2]
    movq                m3, [srcq+src_stride3q]
    movq                m4, [dstq]
    movq                m5, [dstq+dst_strideq*1]
    movq                m6, [dstq+dst_strideq*2]
    movq                m7, [dstq+dst_stride3q]
    W_ABS_DIFF m0, m1, m2, m3, m4, m5, m6, m7
; Don't convert to 32 bit integers: 4*4 abs diffs of 12-bits fits in 16 bits.
; Accumulate onto m0
    %define            sum  m0
    paddw              sum, m1
    paddw               m2, m3
    paddw              sum, m2
; Horizontal reduction
    pshuflw             m1, sum, q2323
    paddw              sum, m1
    pshuflw             m1, sum, q1111
    paddw              sum, m1
    movd               eax, sum
; Convert to 16-bits since the upper half of eax is dirty
    movzx              eax, ax
    RET

%if ARCH_X86_64

; 10-bit only
INIT_XMM ssse3
cglobal sad_8x8_hbd10, 4, 7, 9, src, src_stride, dst, dst_stride, \
                                src_stride3, dst_stride3, cnt
    lea       src_stride3q, [src_strideq*3]
    lea       dst_stride3q, [dst_strideq*3]
    mov               cntd, 2
    %define            sum  m0
    pxor               sum, sum
.loop:
    movu                m1, [srcq]
    movu                m2, [srcq+src_strideq*1]
    movu                m3, [srcq+src_strideq*2]
    movu                m4, [srcq+src_stride3q]
    lea               srcq, [srcq+src_strideq*4]
    movu                m5, [dstq]
    movu                m6, [dstq+dst_strideq*1]
    movu                m7, [dstq+dst_strideq*2]
    movu                m8, [dstq+dst_stride3q]
    lea               dstq, [dstq+dst_strideq*4]
    W_ABS_DIFF m1, m2, m3, m4, m5, m6, m7, m8
    paddw               m1, m2
    paddw               m3, m4
    paddw              sum, m1
    paddw              sum, m3
    dec               cntd
    jg .loop
; Don't convert to 32 bit integers: 8*8 abs diffs of 10-bits fits in 16 bits.
; Horizontal reduction
    movhlps             m1, sum
    paddw              sum, m1
    pshuflw             m1, sum, q2323
    paddw              sum, m1
    pshuflw             m1, sum, q1111
    paddw              sum, m1
    movd               eax, m0
; Convert to 16-bits since the upper half of eax is dirty
    movzx              eax, ax
    RET

INIT_XMM ssse3
cglobal sad_16x16_hbd, 4, 5, 9, src, src_stride, dst, dst_stride, \
                                cnt
    mov               cntd, 8
    %define            sum  m0
    pxor               sum, sum
.loop:
    movu                m1, [srcq]
    movu                m2, [srcq+16]
    movu                m3, [srcq+src_strideq]
    movu                m4, [srcq+src_strideq+16]
    lea               srcq, [srcq+src_strideq*2]
    movu                m5, [dstq]
    movu                m6, [dstq+16]
    movu                m7, [dstq+dst_strideq]
    movu                m8, [dstq+dst_strideq+16]
    lea               dstq, [dstq+dst_strideq*2]
    W_ABS_DIFF m1, m2, m3, m4, m5, m6, m7, m8
    paddw               m1, m2
    paddw               m3, m4
    paddw              sum, m1
    paddw              sum, m3
    dec               cntd
    jg .loop
; Convert to 32-bits
    pxor                m1, m1
    punpcklwd           m2, sum, m1
    punpckhwd          sum, m1
    paddd              sum, m2
; Horizontal reduction
    movhlps             m1, sum
    paddd              sum, m1
    pshufd              m1, sum, q1111
    paddd              sum, m1
    movd               eax, sum
    RET

;10 bit only
INIT_XMM ssse3
cglobal sad_32x32_hbd10, 4, 5, 10, src, src_stride, dst, dst_stride, \
                                   cnt
    mov               cntd, 32
; Accumulate onto multiple registers to avoid overflowing before converting
;   to 32-bits.
    pxor                m0, m0
    pxor                m1, m1
.loop:
    movu                m2, [srcq]
    movu                m3, [srcq+16]
    movu                m4, [srcq+32]
    movu                m5, [srcq+48]
    lea               srcq, [srcq+src_strideq]
    movu                m6, [dstq]
    movu                m7, [dstq+16]
    movu                m8, [dstq+32]
    movu                m9, [dstq+48]
    lea               dstq, [dstq+dst_strideq]
    W_ABS_DIFF m2, m3, m4, m5, m6, m7, m8, m9
    paddw               m2, m3
    paddw               m4, m5
    paddw               m0, m2
    paddw               m1, m4
    dec               cntd
    jg .loop
; Convert to 32-bits
    pxor                m2, m2
    punpcklwd           m3, m0, m2
    punpckhwd           m0, m2
    paddd               m0, m3
    punpcklwd           m3, m1, m2
    punpckhwd           m1, m2
    paddd               m1, m3
    paddd               m0, m1
; Horizontal reduction
    movhlps             m1, m0
    paddd               m0, m1
    pshufd              m1, m0, q1111
    paddd               m0, m1
    movd               eax, m0
    RET

%macro SAD_64X16_HBD10_INTERNAL 1
    mov                 %1, 16
; Accumulate onto multiple registers to avoid overflowing before converting
;   to 32-bits.
; In this case, we need to be able to able to fit into 16-bit SIGNED integers.
    pxor                m1, m1
    pxor                m2, m2
    pxor                m3, m3
    pxor                m4, m4
.innerloop:
    movu                m5, [srcq]
    movu                m6, [srcq+16]
    movu                m7, [srcq+32]
    movu                m8, [srcq+48]
    movu                m9, [dstq]
    movu               m10, [dstq+16]
    movu               m11, [dstq+32]
    movu               m12, [dstq+48]
    W_ABS_DIFF m5, m6, m7, m8, m9, m10, m11, m12
; Evenly distribute abs diffs among the registers we use for accumulation.
    paddw               m1, m5
    paddw               m2, m6
    paddw               m3, m7
    paddw               m4, m8
    movu                m5, [srcq+64]
    movu                m6, [srcq+80]
    movu                m7, [srcq+96]
    movu                m8, [srcq+112]
    lea               srcq, [srcq+src_strideq]
    movu                m9, [dstq+64]
    movu               m10, [dstq+80]
    movu               m11, [dstq+96]
    movu               m12, [dstq+112]
    lea               dstq, [dstq+dst_strideq]
    W_ABS_DIFF m5, m6, m7, m8, m9, m10, m11, m12
; Evenly distribute abs diffs among the registers we use for accumulation.
    paddw               m1, m5
    paddw               m2, m6
    paddw               m3, m7
    paddw               m4, m8
    dec                 %1
    jg .innerloop
; Convert to 32-bits by performing (-1*a) + (-1*b) on pairs of horizontal words.
;   This has to be corrected for later.
; TODO: punpck might be faster since we only have to do it half as much.
    pcmpeqd             m5, m5
    pmaddwd             m1, m5
    pmaddwd             m2, m5
    pmaddwd             m3, m5
    pmaddwd             m4, m5
; Reduce from 4 to 2 regisers, then add them to m0
    paddd               m1, m2
    paddd               m3, m4
    paddd               m0, m1
    paddd               m0, m3
%endmacro

;10 bit only
INIT_XMM ssse3
cglobal sad_64x64_hbd10, 4, 5, 13, src, src_stride, dst, dst_stride, \
                                   cnt1, cnt2
    pxor                m0, m0
; Repeatably accumulates sad from horizontal slices of the block onto m0. Each
;   call increases src and dst as it runs allowing the next call to carry on
;   from where the previous call left off.
; It should be noted that in the process of converting from 16 to 32-bits, the
;   function performs (-1*a) + (-1*b) on pairs of horizontal words. This is
;   corrected for by negating the final output.
    mov               cnt1d, 4
    .loop
    SAD_64X16_HBD10_INTERNAL cnt2d
    dec               cnt1d
    jg .loop
; Horizontal reduction
    movhlps             m1, m0
    paddd               m0, m1
    pshufd              m1, m0, q1111
    paddd               m0, m1
    movd               eax, m0
; Negate reverse the change in sign cause by converting to 32-bits.
    neg                eax
    RET

%macro SAD_128X8_HBD10_INTERNAL 2
    mov                 %1, 8
; Accumulate onto multiple registers to avoid overflowing before converting
;   to 32-bits.
; In this case, we need to be able to able to fit into 16-bit SIGNED integers.
    pxor                m1, m1
    pxor                m2, m2
    pxor                m3, m3
    pxor                m4, m4
.outer_loop:
; Iterate over columns in this row.
    mov                 %2, 4
.inner_loop:
    movu                m5, [srcq]
    movu                m6, [srcq+16]
    movu                m7, [srcq+32]
    movu                m8, [srcq+48]
    lea               srcq, [srcq+64]
    movu                m9, [dstq]
    movu               m10, [dstq+16]
    movu               m11, [dstq+32]
    movu               m12, [dstq+48]
    lea               dstq, [dstq+64]
    W_ABS_DIFF m5, m6, m7, m8, m9, m10, m11, m12
; Evenly distribute abs diffs among the registers we use for accumulation.
    paddw               m1, m5
    paddw               m2, m6
    paddw               m3, m7
    paddw               m4, m8
    dec                 %2
    jg .inner_loop
; When iterating to the next row, subtract the columns we iterated by.
    lea               srcq, [srcq+src_strideq-256]
    lea               dstq, [dstq+dst_strideq-256]
    dec                 %1
    jg .outer_loop
; Convert to 32-bits by performing (-1*a) + (-1*b) on pairs of horizontal words.
;   This has to be corrected for later.
; TODO: punpck might be faster since we only have to do it half as much.
    pcmpeqd             m5, m5
    pmaddwd             m1, m5
    pmaddwd             m2, m5
    pmaddwd             m3, m5
    pmaddwd             m4, m5
; Reduce from 4 to 2 regisers, then add them to m0
    paddd               m1, m2
    paddd               m3, m4
    paddd               m0, m1
    paddd               m0, m3
%endmacro

;10 bit only
INIT_XMM ssse3
cglobal sad_128x128_hbd10, 4, 7, 13, src, src_stride, dst, dst_stride, \
                                     cnt1, cnt2, cnt3
    pxor                m0, m0
; Repeatably accumulates sad from horizontal slices of the block onto m0. Each
;   call increases src and dst as it runs allowing the next call to carry on
;   from where the previous call left off.
; It should be noted that in the process of converting from 16 to 32-bits, the
;   function performs (-1*a) + (-1*b) on pairs of horizontal words. This is
;   corrected for by negating the final output.
    mov              cnt1d, 16
    .loop
    SAD_128X8_HBD10_INTERNAL cnt2d, cnt3d
    dec              cnt1d
    jg .loop
; Horizontal reduction
    movhlps             m1, m0
    paddd               m0, m1
    pshufd              m1, m0, q1111
    paddd               m0, m1
    movd               eax, m0
; Negate reverse the change in sign cause by converting to 32-bits.
    neg                eax
    RET

%endif
