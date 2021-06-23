; Copyright © 2021, VideoLAN and dav1d authors
; Copyright © 2021, Two Orioles, LLC
; All rights reserved.
;
; Redistribution and use in source and binary forms, with or without
; modification, are permitted provided that the following conditions are met:
;
; 1. Redistributions of source code must retain the above copyright notice, this
;    list of conditions and the following disclaimer.
;
; 2. Redistributions in binary form must reproduce the above copyright notice,
;    this list of conditions and the following disclaimer in the documentation
;    and/or other materials provided with the distribution.
;
; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
; ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
; WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
; DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
; ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
; (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
; ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
; (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
; SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION_RODATA

pal_pred_shuf: db  0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15

pb_0_1:  times 4 db 0, 1
pb_2_3:  times 4 db 2, 3
pw_512:  times 4 dw 512
pw_2048: times 4 dw 2048

%macro JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - 2*4)
    %xdefine %%base mangle(private_prefix %+ _%1_%2)
    %%table:
    %rep %0 - 2
        dd %%base %+ .%3 - (%%table - 2*4)
        %rotate 1
    %endrep
%endmacro

%define ipred_dc_splat_16bpc_ssse3_table (ipred_dc_16bpc_ssse3_table + 10*4)
%define ipred_dc_128_16bpc_ssse3_table   (ipred_dc_16bpc_ssse3_table + 15*4)

JMP_TABLE ipred_dc_left_16bpc,    ssse3, h4, h8, h16, h32, h64
JMP_TABLE ipred_dc_16bpc,         ssse3, h4, h8, h16, h32, h64, w4, w8, w16, w32, w64, \
                                         s4-10*4, s8-10*4, s16-10*4, s32-10*4, s64-10*4, \
                                         s4-15*4, s8-15*4, s16c-15*4, s32c-15*4, s64-15*4
JMP_TABLE ipred_h_16bpc,          ssse3, w4, w8, w16, w32, w64
JMP_TABLE pal_pred_16bpc,         ssse3, w4, w8, w16, w32, w64

cextern smooth_weights_1d_16bpc
cextern smooth_weights_2d_16bpc

SECTION .text

%macro REPX 2-*
    %xdefine %%f(x) %1
%rep %0 - 1
    %rotate 1
    %%f(%1)
%endrep
%endmacro

INIT_XMM ssse3
cglobal ipred_dc_top_16bpc, 3, 7, 6, dst, stride, tl, w, h
    LEA                  r5, ipred_dc_left_16bpc_ssse3_table
    movd                 m4, wm
    tzcnt                wd, wm
    add                 tlq, 2
    movifnidn            hd, hm
    pxor                 m3, m3
    pavgw                m4, m3
    movd                 m5, wd
    movu                 m0, [tlq]
    movsxd               r6, [r5+wq*4]
    add                  r6, r5
    add                  r5, ipred_dc_128_16bpc_ssse3_table-ipred_dc_left_16bpc_ssse3_table
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6

cglobal ipred_dc_left_16bpc, 3, 7, 6, dst, stride, tl, w, h, stride3
    LEA                  r5, ipred_dc_left_16bpc_ssse3_table
    mov                  hd, hm
    movd                 m4, hm
    tzcnt               r6d, hd
    sub                 tlq, hq
    tzcnt                wd, wm
    pxor                 m3, m3
    sub                 tlq, hq
    pavgw                m4, m3
    movd                 m5, r6d
    movu                 m0, [tlq]
    movsxd               r6, [r5+r6*4]
    add                  r6, r5
    add                  r5, ipred_dc_128_16bpc_ssse3_table-ipred_dc_left_16bpc_ssse3_table
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6
.h64:
    movu                 m2, [tlq+112]
    movu                 m1, [tlq+ 96]
    paddw                m0, m2
    movu                 m2, [tlq+ 80]
    paddw                m1, m2
    movu                 m2, [tlq+ 64]
    paddw                m0, m2
    paddw                m0, m1
.h32:
    movu                 m1, [tlq+ 48]
    movu                 m2, [tlq+ 32]
    paddw                m1, m2
    paddw                m0, m1
.h16:
    movu                 m1, [tlq+ 16]
    paddw                m0, m1
.h8:
    movhlps              m1, m0
    paddw                m0, m1
.h4:
    punpcklwd            m0, m3
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m4, m0, q1032
    paddd                m0, m4
    psrld                m0, m5
    lea            stride3q, [strideq*3]
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
    jmp                  wq

cglobal ipred_dc_16bpc, 4, 7, 6, dst, stride, tl, w, h, stride3
    movifnidn            hd, hm
    tzcnt               r6d, hd
    lea                 r5d, [wq+hq]
    movd                 m4, r5d
    tzcnt               r5d, r5d
    movd                 m5, r5d
    LEA                  r5, ipred_dc_16bpc_ssse3_table
    tzcnt                wd, wd
    movsxd               r6, [r5+r6*4]
    movsxd               wq, [r5+wq*4+5*4]
    pxor                 m3, m3
    psrlw                m4, 1
    add                  r6, r5
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  r6
.h4:
    movq                 m0, [tlq-8]
    jmp                  wq
.w4:
    movq                 m1, [tlq+2]
    paddw                m1, m0
    punpckhwd            m0, m3
    punpcklwd            m1, m3
    paddd                m0, m1
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m1, m0, q1032
    paddd                m0, m1
    cmp                  hd, 4
    jg .w4_mul
    psrlw                m0, 3
    jmp .w4_end
.w4_mul:
    mov                 r2d, 0xAAAB
    mov                 r3d, 0x6667
    cmp                  hd, 16
    cmove               r2d, r3d
    psrld                m0, 2
    movd                 m1, r2d
    pmulhuw              m0, m1
    psrlw                m0, 1
.w4_end:
    pshuflw              m0, m0, q0000
.s4:
    movq   [dstq+strideq*0], m0
    movq   [dstq+strideq*1], m0
    movq   [dstq+strideq*2], m0
    movq   [dstq+stride3q ], m0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s4
    RET
.h8:
    mova                 m0, [tlq-16]
    jmp                  wq
.w8:
    movu                 m1, [tlq+2]
    paddw                m0, m1
    punpcklwd            m1, m0, m3
    punpckhwd            m0, m3
    paddd                m0, m1
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m1, m0, q1032
    paddd                m0, m1
    psrld                m0, m5
    cmp                  hd, 8
    je .w8_end
    mov                 r2d, 0xAAAB
    mov                 r3d, 0x6667
    cmp                  hd, 32
    cmove               r2d, r3d
    movd                 m1, r2d
    pmulhuw              m0, m1
    psrlw                m0, 1
.w8_end:
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
.s8:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m0
    mova   [dstq+strideq*2], m0
    mova   [dstq+stride3q ], m0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s8
    RET
.h16:
    mova                 m0, [tlq-32]
    paddw                m0, [tlq-16]
    jmp                  wq
.w16:
    movu                 m1, [tlq+ 2]
    movu                 m2, [tlq+18]
    paddw                m1, m2
    paddw                m0, m1
    punpckhwd            m1, m0, m3
    punpcklwd            m0, m3
    paddd                m0, m1
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m1, m0, q1032
    paddd                m0, m1
    psrld                m0, m5
    cmp                  hd, 16
    je .w16_end
    mov                 r2d, 0xAAAB
    mov                 r3d, 0x6667
    test                 hd, 8|32
    cmovz               r2d, r3d
    movd                 m1, r2d
    pmulhuw              m0, m1
    psrlw                m0, 1
.w16_end:
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
.s16c:
    mova                 m1, m0
.s16:
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m1
    mova [dstq+strideq*1+16*0], m0
    mova [dstq+strideq*1+16*1], m1
    mova [dstq+strideq*2+16*0], m0
    mova [dstq+strideq*2+16*1], m1
    mova [dstq+stride3q +16*0], m0
    mova [dstq+stride3q +16*1], m1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s16
    RET
.h32:
    mova                 m0, [tlq-64]
    paddw                m0, [tlq-48]
    paddw                m0, [tlq-32]
    paddw                m0, [tlq-16]
    jmp                  wq
.w32:
    movu                 m1, [tlq+ 2]
    movu                 m2, [tlq+18]
    paddw                m1, m2
    movu                 m2, [tlq+34]
    paddw                m0, m2
    movu                 m2, [tlq+50]
    paddw                m1, m2
    paddw                m0, m1
    punpcklwd            m1, m0, m3
    punpckhwd            m0, m3
    paddd                m0, m1
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m1, m0, q1032
    paddd                m0, m1
    psrld                m0, m5
    cmp                  hd, 32
    je .w32_end
    mov                 r2d, 0xAAAB
    mov                 r3d, 0x6667
    cmp                  hd, 8
    cmove               r2d, r3d
    movd                 m1, r2d
    pmulhuw              m0, m1
    psrlw                m0, 1
.w32_end:
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
.s32c:
    mova                 m1, m0
    mova                 m2, m0
    mova                 m3, m0
.s32:
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m1
    mova [dstq+strideq*0+16*2], m2
    mova [dstq+strideq*0+16*3], m3
    mova [dstq+strideq*1+16*0], m0
    mova [dstq+strideq*1+16*1], m1
    mova [dstq+strideq*1+16*2], m2
    mova [dstq+strideq*1+16*3], m3
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .s32
    RET
.h64:
    mova                 m0, [tlq-128]
    mova                 m1, [tlq-112]
    paddw                m0, [tlq- 96]
    paddw                m1, [tlq- 80]
    paddw                m0, [tlq- 64]
    paddw                m1, [tlq- 48]
    paddw                m0, [tlq- 32]
    paddw                m1, [tlq- 16]
    paddw                m0, m1
    jmp                  wq
.w64:
    movu                 m1, [tlq+  2]
    movu                 m2, [tlq+ 18]
    paddw                m1, m2
    movu                 m2, [tlq+ 34]
    paddw                m0, m2
    movu                 m2, [tlq+ 50]
    paddw                m1, m2
    movu                 m2, [tlq+ 66]
    paddw                m0, m2
    movu                 m2, [tlq+ 82]
    paddw                m1, m2
    movu                 m2, [tlq+ 98]
    paddw                m0, m2
    movu                 m2, [tlq+114]
    paddw                m1, m2
    paddw                m0, m1
    punpcklwd            m1, m0, m3
    punpckhwd            m0, m3
    paddd                m0, m1
    paddd                m4, m0
    punpckhqdq           m0, m0
    paddd                m0, m4
    pshuflw              m1, m0, q1032
    paddd                m0, m1
    psrld                m0, m5
    cmp                  hd, 64
    je .w64_end
    mov                 r2d, 0xAAAB
    mov                 r3d, 0x6667
    cmp                  hd, 16
    cmove               r2d, r3d
    movd                 m1, r2d
    pmulhuw              m0, m1
    psrlw                m0, 1
.w64_end:
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
.s64:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m0
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m0
    mova        [dstq+16*4], m0
    mova        [dstq+16*5], m0
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m0
    add                dstq, strideq
    dec                  hd
    jg .s64
    RET

cglobal ipred_dc_128_16bpc, 2, 7, 6, dst, stride, tl, w, h, stride3
    mov                 r6d, r8m
    LEA                  r5, ipred_dc_128_16bpc_ssse3_table
    tzcnt                wd, wm
    shr                 r6d, 11
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    movddup              m0, [r5-ipred_dc_128_16bpc_ssse3_table+pw_512+r6*8]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq

cglobal ipred_v_16bpc, 4, 7, 6, dst, stride, tl, w, h, stride3
    LEA                  r5, ipred_dc_splat_16bpc_ssse3_table
    movifnidn            hd, hm
    movu                 m0, [tlq+  2]
    movu                 m1, [tlq+ 18]
    movu                 m2, [tlq+ 34]
    movu                 m3, [tlq+ 50]
    cmp                  wd, 64
    je .w64
    tzcnt                wd, wd
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq
.w64:
    WIN64_SPILL_XMM 8
    movu                 m4, [tlq+ 66]
    movu                 m5, [tlq+ 82]
    movu                 m6, [tlq+ 98]
    movu                 m7, [tlq+114]
.w64_loop:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    mova        [dstq+16*2], m2
    mova        [dstq+16*3], m3
    mova        [dstq+16*4], m4
    mova        [dstq+16*5], m5
    mova        [dstq+16*6], m6
    mova        [dstq+16*7], m7
    add                dstq, strideq
    dec                  hd
    jg .w64_loop
    RET

cglobal ipred_h_16bpc, 3, 6, 4, dst, stride, tl, w, h, stride3
%define base r5-ipred_h_16bpc_ssse3_table
    tzcnt                wd, wm
    LEA                  r5, ipred_h_16bpc_ssse3_table
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    movddup              m2, [base+pb_0_1]
    movddup              m3, [base+pb_2_3]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq
.w4:
    sub                 tlq, 8
    movq                 m3, [tlq]
    pshuflw              m0, m3, q3333
    pshuflw              m1, m3, q2222
    pshuflw              m2, m3, q1111
    pshuflw              m3, m3, q0000
    movq   [dstq+strideq*0], m0
    movq   [dstq+strideq*1], m1
    movq   [dstq+strideq*2], m2
    movq   [dstq+stride3q ], m3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w4
    RET
.w8:
    sub                 tlq, 8
    movq                 m3, [tlq]
    punpcklwd            m3, m3
    pshufd               m0, m3, q3333
    pshufd               m1, m3, q2222
    pshufd               m2, m3, q1111
    pshufd               m3, m3, q0000
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    mova   [dstq+strideq*2], m2
    mova   [dstq+stride3q ], m3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w8
    RET
.w16:
    sub                 tlq, 4
    movd                 m1, [tlq]
    pshufb               m0, m1, m3
    pshufb               m1, m2
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m0
    mova [dstq+strideq*1+16*0], m1
    mova [dstq+strideq*1+16*1], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16
    RET
.w32:
    sub                 tlq, 4
    movd                 m1, [tlq]
    pshufb               m0, m1, m3
    pshufb               m1, m2
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m0
    mova [dstq+strideq*0+16*2], m0
    mova [dstq+strideq*0+16*3], m0
    mova [dstq+strideq*1+16*0], m1
    mova [dstq+strideq*1+16*1], m1
    mova [dstq+strideq*1+16*2], m1
    mova [dstq+strideq*1+16*3], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w32
    RET
.w64:
    sub                 tlq, 2
    movd                 m0, [tlq]
    pshufb               m0, m2
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m0
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m0
    mova        [dstq+16*4], m0
    mova        [dstq+16*5], m0
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m0
    add                dstq, strideq
    dec                  hd
    jg .w64
    RET

cglobal ipred_paeth_16bpc, 4, 6, 8, dst, stride, tl, w, h, left
%define base r5-ipred_paeth_16bpc_ssse3_table
    movifnidn            hd, hm
    pshuflw              m4, [tlq], q0000
    mov               leftq, tlq
    add                  hd, hd
    punpcklqdq           m4, m4      ; topleft
    sub               leftq, hq
    and                  wd, ~7
    jnz .w8
    movddup              m5, [tlq+2] ; top
    psubw                m6, m5, m4
    pabsw                m7, m6
.w4_loop:
    movd                 m1, [leftq+hq-4]
    punpcklwd            m1, m1
    punpckldq            m1, m1      ; left
%macro PAETH 0
    paddw                m0, m6, m1
    psubw                m2, m4, m0  ; tldiff
    psubw                m0, m5      ; tdiff
    pabsw                m2, m2
    pabsw                m0, m0
    pminsw               m2, m0
    pcmpeqw              m0, m2
    pand                 m3, m5, m0
    pandn                m0, m4
    por                  m0, m3
    pcmpgtw              m3, m7, m2
    pand                 m0, m3
    pandn                m3, m1
    por                  m0, m3
%endmacro
    PAETH
    movhps [dstq+strideq*0], m0
    movq   [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2*2
    jg .w4_loop
    RET
.w8:
%if ARCH_X86_32
    PUSH                 r6
    %define             r7d  hm
    %assign regs_used     7
%elif WIN64
    movaps              r4m, m8
    PUSH                 r7
    %assign regs_used     8
%endif
%if ARCH_X86_64
    movddup              m8, [pb_0_1]
%endif
    lea                 tlq, [tlq+wq*2+2]
    neg                  wq
    mov                 r7d, hd
.w8_loop0:
    movu                 m5, [tlq+wq*2]
    mov                  r6, dstq
    add                dstq, 16
    psubw                m6, m5, m4
    pabsw                m7, m6
.w8_loop:
    movd                 m1, [leftq+hq-2]
%if ARCH_X86_64
    pshufb               m1, m8
%else
    pshuflw              m1, m1, q0000
    punpcklqdq           m1, m1
%endif
    PAETH
    mova               [r6], m0
    add                  r6, strideq
    sub                  hd, 1*2
    jg .w8_loop
    mov                  hd, r7d
    add                  wq, 8
    jl .w8_loop0
%if WIN64
    movaps               m8, r4m
%endif
    RET

%if ARCH_X86_64
DECLARE_REG_TMP 7
%else
DECLARE_REG_TMP 4
%endif

cglobal ipred_smooth_v_16bpc, 4, 6, 6, dst, stride, tl, w, h, weights
    LEA            weightsq, smooth_weights_1d_16bpc
    mov                  hd, hm
    lea            weightsq, [weightsq+hq*4]
    neg                  hq
    movd                 m5, [tlq+hq*2] ; bottom
    pshuflw              m5, m5, q0000
    punpcklqdq           m5, m5
    cmp                  wd, 4
    jne .w8
    movddup              m4, [tlq+2]    ; top
    lea                  r3, [strideq*3]
    psubw                m4, m5         ; top - bottom
.w4_loop:
    movq                 m1, [weightsq+hq*2]
    punpcklwd            m1, m1
    pshufd               m0, m1, q1100
    punpckhdq            m1, m1
    pmulhrsw             m0, m4
    pmulhrsw             m1, m4
    paddw                m0, m5
    paddw                m1, m5
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movq   [dstq+strideq*2], m1
    movhps [dstq+r3       ], m1
    lea                dstq, [dstq+strideq*4]
    add                  hq, 4
    jl .w4_loop
    RET
.w8:
%if ARCH_X86_32
    PUSH                 r6
    %assign regs_used     7
    mov                  hm, hq
    %define              hq  hm
%elif WIN64
    PUSH                 r7
    %assign regs_used     8
%endif
.w8_loop0:
    mov                  t0, hq
    movu                 m4, [tlq+2]
    add                 tlq, 16
    mov                  r6, dstq
    add                dstq, 16
    psubw                m4, m5
.w8_loop:
    movq                 m3, [weightsq+t0*2]
    punpcklwd            m3, m3
    pshufd               m0, m3, q0000
    pshufd               m1, m3, q1111
    pshufd               m2, m3, q2222
    pshufd               m3, m3, q3333
    REPX   {pmulhrsw x, m4}, m0, m1, m2, m3
    REPX   {paddw    x, m5}, m0, m1, m2, m3
    mova     [r6+strideq*0], m0
    mova     [r6+strideq*1], m1
    lea                  r6, [r6+strideq*2]
    mova     [r6+strideq*0], m2
    mova     [r6+strideq*1], m3
    lea                  r6, [r6+strideq*2]
    add                  t0, 4
    jl .w8_loop
    sub                  wd, 8
    jg .w8_loop0
    RET

cglobal ipred_smooth_h_16bpc, 3, 6, 6, dst, stride, tl, w, h, weights
    LEA            weightsq, smooth_weights_1d_16bpc
    mov                  wd, wm
    movifnidn            hd, hm
    movd                 m5, [tlq+wq*2] ; right
    sub                 tlq, 8
    add                  hd, hd
    pshuflw              m5, m5, q0000
    sub                 tlq, hq
    punpcklqdq           m5, m5
    cmp                  wd, 4
    jne .w8
    movddup              m4, [weightsq+4*2]
    lea                  r3, [strideq*3]
.w4_loop:
    movq                 m1, [tlq+hq]   ; left
    punpcklwd            m1, m1
    psubw                m1, m5         ; left - right
    pshufd               m0, m1, q3322
    punpckldq            m1, m1
    pmulhrsw             m0, m4
    pmulhrsw             m1, m4
    paddw                m0, m5
    paddw                m1, m5
    movhps [dstq+strideq*0], m0
    movq   [dstq+strideq*1], m0
    movhps [dstq+strideq*2], m1
    movq   [dstq+r3       ], m1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4*2
    jg .w4_loop
    RET
.w8:
    lea            weightsq, [weightsq+wq*4]
    neg                  wq
%if ARCH_X86_32
    PUSH                 r6
    %assign regs_used     7
    %define              hd  hm
%elif WIN64
    PUSH                 r7
    %assign regs_used     8
%endif
.w8_loop0:
    mov                 t0d, hd
    mova                 m4, [weightsq+wq*2]
    mov                  r6, dstq
    add                dstq, 16
.w8_loop:
    movq                 m3, [tlq+t0*(1+ARCH_X86_32)]
    punpcklwd            m3, m3
    psubw                m3, m5
    pshufd               m0, m3, q3333
    pshufd               m1, m3, q2222
    pshufd               m2, m3, q1111
    pshufd               m3, m3, q0000
    REPX   {pmulhrsw x, m4}, m0, m1, m2, m3
    REPX   {paddw    x, m5}, m0, m1, m2, m3
    mova     [r6+strideq*0], m0
    mova     [r6+strideq*1], m1
    lea                  r6, [r6+strideq*2]
    mova     [r6+strideq*0], m2
    mova     [r6+strideq*1], m3
    lea                  r6, [r6+strideq*2]
    sub                 t0d, 4*(1+ARCH_X86_64)
    jg .w8_loop
    add                  wq, 8
    jl .w8_loop0
    RET

%if ARCH_X86_64
DECLARE_REG_TMP 10
%else
DECLARE_REG_TMP 3
%endif

cglobal ipred_smooth_16bpc, 3, 7, 8, dst, stride, tl, w, h, \
                                     h_weights, v_weights, top
    LEA          h_weightsq, smooth_weights_2d_16bpc
    mov                  wd, wm
    mov                  hd, hm
    movd                 m7, [tlq+wq*2] ; right
    lea          v_weightsq, [h_weightsq+hq*8]
    neg                  hq
    movd                 m6, [tlq+hq*2] ; bottom
    pshuflw              m7, m7, q0000
    pshuflw              m6, m6, q0000
    cmp                  wd, 4
    jne .w8
    movq                 m4, [tlq+2]    ; top
    mova                 m5, [h_weightsq+4*4]
    punpcklwd            m4, m6         ; top, bottom
    pxor                 m6, m6
.w4_loop:
    movq                 m1, [v_weightsq+hq*4]
    sub                 tlq, 4
    movd                 m3, [tlq]      ; left
    pshufd               m0, m1, q0000
    pshufd               m1, m1, q1111
    pmaddwd              m0, m4
    punpcklwd            m3, m7         ; left, right
    pmaddwd              m1, m4
    pshufd               m2, m3, q1111
    pshufd               m3, m3, q0000
    pmaddwd              m2, m5
    pmaddwd              m3, m5
    paddd                m0, m2
    paddd                m1, m3
    psrld                m0, 8
    psrld                m1, 8
    packssdw             m0, m1
    pavgw                m0, m6
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    add                  hq, 2
    jl .w4_loop
    RET
.w8:
%if ARCH_X86_32
    lea          h_weightsq, [h_weightsq+wq*4]
    mov                  t0, tlq
    mov                 r1m, tlq
    mov                 r2m, hq
    %define              m8  [h_weightsq+16*0]
    %define              m9  [h_weightsq+16*1]
%else
%if WIN64
    movaps              r4m, m8
    movaps              r6m, m9
    PUSH                 r7
    PUSH                 r8
%endif
    PUSH                 r9
    PUSH                r10
    %assign       regs_used  11
    lea          h_weightsq, [h_weightsq+wq*8]
    lea                topq, [tlq+wq*2]
    neg                  wq
    mov                  r8, tlq
    mov                  r9, hq
%endif
    punpcklqdq           m6, m6
.w8_loop0:
%if ARCH_X86_32
    movu                 m5, [t0+2]
    add                  t0, 16
    mov                 r0m, t0
%else
    movu                 m5, [topq+wq*2+2]
    mova                 m8, [h_weightsq+wq*4+16*0]
    mova                 m9, [h_weightsq+wq*4+16*1]
%endif
    mov                  t0, dstq
    add                dstq, 16
    punpcklwd            m4, m5, m6
    punpckhwd            m5, m6
.w8_loop:
    movd                 m1, [v_weightsq+hq*4]
    sub                 tlq, 2
    movd                 m3, [tlq]      ; left
    pshufd               m1, m1, q0000
    pmaddwd              m0, m4, m1
    pshuflw              m3, m3, q0000
    pmaddwd              m1, m5
    punpcklwd            m3, m7         ; left, right
    pmaddwd              m2, m8, m3
    pmaddwd              m3, m9
    paddd                m0, m2
    paddd                m1, m3
    psrld                m0, 8
    psrld                m1, 8
    packssdw             m0, m1
    pxor                 m1, m1
    pavgw                m0, m1
    mova               [t0], m0
    add                  t0, strideq
    inc                  hq
    jl .w8_loop
%if ARCH_X86_32
    mov                  t0, r0m
    mov                 tlq, r1m
    add          h_weightsq, 16*2
    mov                  hq, r2m
    sub            dword wm, 8
    jg .w8_loop0
%else
    mov                 tlq, r8
    mov                  hq, r9
    add                  wq, 8
    jl .w8_loop0
%endif
%if WIN64
    movaps               m8, r4m
    movaps               m9, r6m
%endif
    RET

cglobal pal_pred_16bpc, 4, 5, 5, dst, stride, pal, idx, w, h
%define base r2-pal_pred_16bpc_ssse3_table
%if ARCH_X86_32
    %define              hd  r2d
%endif
    mova                 m3, [palq]
    LEA                  r2, pal_pred_16bpc_ssse3_table
    tzcnt                wd, wm
    pshufb               m3, [base+pal_pred_shuf]
    movsxd               wq, [r2+wq*4]
    pshufd               m4, m3, q1032
    add                  wq, r2
    movifnidn            hd, hm
    jmp                  wq
.w4:
    mova                 m0, [idxq]
    add                idxq, 16
    pshufb               m1, m3, m0
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    movq   [dstq+strideq*0], m1
    movhps [dstq+strideq*1], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 4
    jg .w4
    RET
.w8:
    mova                 m0, [idxq]
    add                idxq, 16
    pshufb               m1, m3, m0
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w8
    RET
.w16:
    mova                 m0, [idxq]
    add                idxq, 16
    pshufb               m1, m3, m0
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    add                dstq, strideq
    dec                  hd
    jg .w16
    RET
.w32:
    mova                 m0, [idxq+16*0]
    pshufb               m1, m3, m0
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova                 m2, [idxq+16*1]
    add                idxq, 16*2
    mova        [dstq+16*0], m0
    pshufb               m0, m3, m2
    mova        [dstq+16*1], m1
    pshufb               m1, m4, m2
    punpcklbw            m2, m0, m1
    punpckhbw            m0, m1
    mova        [dstq+16*2], m2
    mova        [dstq+16*3], m0
    add                dstq, strideq
    dec                  hd
    jg .w32
    RET
.w64:
    mova                 m0, [idxq+16*0]
    pshufb               m1, m3, m0
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova                 m2, [idxq+16*1]
    mova        [dstq+16*0], m0
    pshufb               m0, m3, m2
    mova        [dstq+16*1], m1
    pshufb               m1, m4, m2
    punpcklbw            m2, m0, m1
    punpckhbw            m0, m1
    mova                 m1, [idxq+16*2]
    mova        [dstq+16*2], m2
    pshufb               m2, m3, m1
    mova        [dstq+16*3], m0
    pshufb               m0, m4, m1
    punpcklbw            m1, m2, m0
    punpckhbw            m2, m0
    mova                 m0, [idxq+16*3]
    add                idxq, 16*4
    mova        [dstq+16*4], m1
    pshufb               m1, m3, m0
    mova        [dstq+16*5], m2
    pshufb               m2, m4, m0
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m1
    add                dstq, strideq
    dec                  hd
    jg .w64
    RET
