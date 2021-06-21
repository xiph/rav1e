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

SECTION .text

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
