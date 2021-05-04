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

%if ARCH_X86_64

SECTION_RODATA

pw_512:  times 2 dw 512
pw_2048: times 2 dw 2048

%macro JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - 2*4)
    %xdefine %%base mangle(private_prefix %+ _%1_%2)
    %%table:
    %rep %0 - 2
        dd %%base %+ .%3 - (%%table - 2*4)
        %rotate 1
    %endrep
%endmacro

%define ipred_dc_splat_16bpc_avx2_table (ipred_dc_16bpc_avx2_table + 10*4)

JMP_TABLE ipred_dc_16bpc,         avx2, h4, h8, h16, h32, h64, w4, w8, w16, w32, w64, \
                                        s4-10*4, s8-10*4, s16-10*4, s32-10*4, s64-10*4
JMP_TABLE ipred_dc_left_16bpc,    avx2, h4, h8, h16, h32, h64
JMP_TABLE ipred_h_16bpc,          avx2, w4, w8, w16, w32, w64

SECTION .text

INIT_YMM avx2

cglobal ipred_dc_top_16bpc, 3, 7, 6, dst, stride, tl, w, h
    movifnidn            hd, hm
    add                 tlq, 2
    movd                xm4, wd
    pxor                xm3, xm3
    pavgw               xm4, xm3
    tzcnt                wd, wd
    movd                xm5, wd
    movu                 m0, [tlq]
    lea                  r5, [ipred_dc_left_16bpc_avx2_table]
    movsxd               r6, [r5+wq*4]
    add                  r6, r5
    add                  r5, ipred_dc_splat_16bpc_avx2_table-ipred_dc_left_16bpc_avx2_table
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6

cglobal ipred_dc_left_16bpc, 3, 7, 6, dst, stride, tl, w, h, stride3
    mov                  hd, hm
    sub                 tlq, hq
    movd                xm4, hd
    sub                 tlq, hq
    pxor                xm3, xm3
    pavgw               xm4, xm3
    tzcnt               r6d, hd
    movd                xm5, r6d
    movu                 m0, [tlq]
    lea                  r5, [ipred_dc_left_16bpc_avx2_table]
    movsxd               r6, [r5+r6*4]
    add                  r6, r5
    add                  r5, ipred_dc_splat_16bpc_avx2_table-ipred_dc_left_16bpc_avx2_table
    tzcnt                wd, wd
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6
.h64:
    paddw                m0, [tlq+96]
    paddw                m0, [tlq+64]
.h32:
    paddw                m0, [tlq+32]
.h16:
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
.h8:
    psrldq              xm1, xm0, 8
    paddw               xm0, xm1
.h4:
    punpcklwd           xm0, xm3
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    paddd               xm0, xm4
    psrld               xm0, xm5
    lea            stride3q, [strideq*3]
    vpbroadcastw         m0, xm0
    mova                 m1, m0
    mova                 m2, m0
    mova                 m3, m0
    jmp                  wq

cglobal ipred_dc_16bpc, 3, 7, 6, dst, stride, tl, w, h, stride3
    movifnidn            hd, hm
    tzcnt               r6d, hd
    lea                 r5d, [wq+hq]
    movd                xm4, r5d
    tzcnt               r5d, r5d
    movd                xm5, r5d
    lea                  r5, [ipred_dc_16bpc_avx2_table]
    tzcnt                wd, wd
    movsxd               r6, [r5+r6*4]
    movsxd               wq, [r5+wq*4+5*4]
    pxor                 m3, m3
    psrlw               xm4, 1
    add                  r6, r5
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  r6
.h4:
    movq                xm0, [tlq-8]
    jmp                  wq
.w4:
    movq                xm1, [tlq+2]
    paddw                m0, m4
    paddw                m0, m1
    psrlq                m1, m0, 32
    paddw                m0, m1
    psrld                m1, m0, 16
    paddw                m0, m1
    cmp                  hd, 4
    jg .w4_mul
    psrlw               xm0, 3
    jmp .w4_end
.w4_mul:
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
    lea                 r2d, [hq*2]
    mov                 r6d, 0xAAAB6667
    shrx                r6d, r6d, r2d
    punpckhwd           xm1, xm0, xm3
    punpcklwd           xm0, xm3
    paddd               xm0, xm1
    movd                xm1, r6d
    psrld               xm0, 2
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w4_end:
    vpbroadcastw        xm0, xm0
.s4:
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm0
    movq   [dstq+strideq*2], xm0
    movq   [dstq+stride3q ], xm0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s4
    RET
ALIGN function_align
.h8:
    mova                xm0, [tlq-16]
    jmp                  wq
.w8:
    vextracti128        xm1, m0, 1
    paddw               xm0, [tlq+2]
    paddw               xm0, xm4
    paddw               xm0, xm1
    psrld               xm1, xm0, 16
    paddw               xm0, xm1
    pblendw             xm0, xm3, 0xAA
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    psrld               xm0, xm5
    cmp                  hd, 8
    je .w8_end
    mov                 r6d, 0xAAAB
    mov                 r2d, 0x6667
    cmp                  hd, 32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w8_end:
    vpbroadcastw        xm0, xm0
.s8:
    mova   [dstq+strideq*0], xm0
    mova   [dstq+strideq*1], xm0
    mova   [dstq+strideq*2], xm0
    mova   [dstq+stride3q ], xm0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s8
    RET
ALIGN function_align
.h16:
    mova                 m0, [tlq-32]
    jmp                  wq
.w16:
    paddw                m0, [tlq+2]
    vextracti128        xm1, m0, 1
    paddw               xm0, xm4
    paddw               xm0, xm1
    punpckhwd           xm1, xm0, xm3
    punpcklwd           xm0, xm3
    paddd               xm0, xm1
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    psrld               xm0, xm5
    cmp                  hd, 16
    je .w16_end
    mov                 r6d, 0xAAAB
    mov                 r2d, 0x6667
    test                 hb, 8|32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w16_end:
    vpbroadcastw         m0, xm0
.s16:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m0
    mova   [dstq+strideq*2], m0
    mova   [dstq+stride3q ], m0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s16
    RET
ALIGN function_align
.h32:
    mova                 m0, [tlq-64]
    paddw                m0, [tlq-32]
    jmp                  wq
.w32:
    paddw                m0, [tlq+ 2]
    paddw                m0, [tlq+34]
    vextracti128        xm1, m0, 1
    paddw               xm0, xm4
    paddw               xm0, xm1
    punpcklwd           xm1, xm0, xm3
    punpckhwd           xm0, xm3
    paddd               xm0, xm1
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    psrld               xm0, xm5
    cmp                  hd, 32
    je .w32_end
    lea                 r2d, [hq*2]
    mov                 r6d, 0x6667AAAB
    shrx                r6d, r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w32_end:
    vpbroadcastw         m0, xm0
    mova                 m1, m0
.s32:
    mova [dstq+strideq*0+32*0], m0
    mova [dstq+strideq*0+32*1], m1
    mova [dstq+strideq*1+32*0], m0
    mova [dstq+strideq*1+32*1], m1
    mova [dstq+strideq*2+32*0], m0
    mova [dstq+strideq*2+32*1], m1
    mova [dstq+stride3q +32*0], m0
    mova [dstq+stride3q +32*1], m1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s32
    RET
ALIGN function_align
.h64:
    mova                 m0, [tlq-128]
    mova                 m1, [tlq- 96]
    paddw                m0, [tlq- 64]
    paddw                m1, [tlq- 32]
    paddw                m0, m1
    jmp                  wq
.w64:
    movu                 m1, [tlq+ 2]
    paddw                m0, [tlq+34]
    paddw                m1, [tlq+66]
    paddw                m0, [tlq+98]
    paddw                m0, m1
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
    punpcklwd           xm1, xm0, xm3
    punpckhwd           xm0, xm3
    paddd               xm1, xm4
    paddd               xm0, xm1
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    psrld               xm0, xm5
    cmp                  hd, 64
    je .w64_end
    mov                 r6d, 0x6667AAAB
    shrx                r6d, r6d, hd
    movd                xm1, r6d
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w64_end:
    vpbroadcastw         m0, xm0
    mova                 m1, m0
    mova                 m2, m0
    mova                 m3, m0
.s64:
    mova [dstq+strideq*0+32*0], m0
    mova [dstq+strideq*0+32*1], m1
    mova [dstq+strideq*0+32*2], m2
    mova [dstq+strideq*0+32*3], m3
    mova [dstq+strideq*1+32*0], m0
    mova [dstq+strideq*1+32*1], m1
    mova [dstq+strideq*1+32*2], m2
    mova [dstq+strideq*1+32*3], m3
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .s64
    RET

cglobal ipred_dc_128_16bpc, 2, 7, 6, dst, stride, tl, w, h, stride3
    mov                 r6d, r8m
    shr                 r6d, 11
    lea                  r5, [ipred_dc_splat_16bpc_avx2_table]
    tzcnt                wd, wd
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    vpbroadcastd         m0, [r5-ipred_dc_splat_16bpc_avx2_table+pw_512+r6*4]
    mova                 m1, m0
    mova                 m2, m0
    mova                 m3, m0
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq

cglobal ipred_v_16bpc, 3, 7, 6, dst, stride, tl, w, h, stride3
    movifnidn            hd, hm
    movu                 m0, [tlq+ 2]
    movu                 m1, [tlq+34]
    movu                 m2, [tlq+66]
    movu                 m3, [tlq+98]
    lea                  r5, [ipred_dc_splat_16bpc_avx2_table]
    tzcnt                wd, wd
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq

%macro IPRED_H 2 ; w, store_type
    vpbroadcastw         m0, [tlq-2]
    vpbroadcastw         m1, [tlq-4]
    vpbroadcastw         m2, [tlq-6]
    vpbroadcastw         m3, [tlq-8]
    sub                 tlq, 8
    mov%2  [dstq+strideq*0], m0
    mov%2  [dstq+strideq*1], m1
    mov%2  [dstq+strideq*2], m2
    mov%2  [dstq+stride3q ], m3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w%1
    RET
ALIGN function_align
%endmacro

cglobal ipred_h_16bpc, 3, 6, 4, dst, stride, tl, w, h, stride3
    movifnidn            hd, hm
    lea                  r5, [ipred_h_16bpc_avx2_table]
    tzcnt                wd, wd
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq
INIT_XMM avx2
.w4:
    IPRED_H               4, q
.w8:
    IPRED_H               8, a
INIT_YMM avx2
.w16:
    IPRED_H              16, a
.w32:
    vpbroadcastw         m0, [tlq-2]
    vpbroadcastw         m1, [tlq-4]
    vpbroadcastw         m2, [tlq-6]
    vpbroadcastw         m3, [tlq-8]
    sub                 tlq, 8
    mova [dstq+strideq*0+32*0], m0
    mova [dstq+strideq*0+32*1], m0
    mova [dstq+strideq*1+32*0], m1
    mova [dstq+strideq*1+32*1], m1
    mova [dstq+strideq*2+32*0], m2
    mova [dstq+strideq*2+32*1], m2
    mova [dstq+stride3q +32*0], m3
    mova [dstq+stride3q +32*1], m3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w32
    RET
.w64:
    vpbroadcastw         m0, [tlq-2]
    vpbroadcastw         m1, [tlq-4]
    sub                 tlq, 4
    mova [dstq+strideq*0+32*0], m0
    mova [dstq+strideq*0+32*1], m0
    mova [dstq+strideq*0+32*2], m0
    mova [dstq+strideq*0+32*3], m0
    mova [dstq+strideq*1+32*0], m1
    mova [dstq+strideq*1+32*1], m1
    mova [dstq+strideq*1+32*2], m1
    mova [dstq+strideq*1+32*3], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w64
    RET

%endif
