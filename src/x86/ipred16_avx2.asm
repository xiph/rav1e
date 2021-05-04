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

SECTION_RODATA 32

%macro SMOOTH_WEIGHT_TABLE 1-*
    %rep %0
        dw %1, 256-%1
        %rotate 1
    %endrep
%endmacro

; sm_weights[], but modified to precalculate x and 256-x
smooth_weights: SMOOTH_WEIGHT_TABLE         \
      0,   0, 255, 128, 255, 149,  85,  64, \
    255, 197, 146, 105,  73,  50,  37,  32, \
    255, 225, 196, 170, 145, 123, 102,  84, \
     68,  54,  43,  33,  26,  20,  17,  16, \
    255, 240, 225, 210, 196, 182, 169, 157, \
    145, 133, 122, 111, 101,  92,  83,  74, \
     66,  59,  52,  45,  39,  34,  29,  25, \
     21,  17,  14,  12,  10,   9,   8,   8, \
    255, 248, 240, 233, 225, 218, 210, 203, \
    196, 189, 182, 176, 169, 163, 156, 150, \
    144, 138, 133, 127, 121, 116, 111, 106, \
    101,  96,  91,  86,  82,  77,  73,  69, \
     65,  61,  57,  54,  50,  47,  44,  41, \
     38,  35,  32,  29,  27,  25,  22,  20, \
     18,  16,  15,  13,  12,  10,   9,   8, \
      7,   6,   6,   5,   5,   4,   4,   4

ipred_hv_shuf: db  6,  7,  6,  7,  0,  1,  2,  3,  2,  3,  2,  3,  8,  9, 10, 11
               db  4,  5,  4,  5,  4,  5,  6,  7,  0,  1,  0,  1, 12, 13, 14, 15

pw_512:  times 2 dw 512
pw_2048: times 2 dw 2048
pd_128:  dd 128
pd_256:  dd 256

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
JMP_TABLE ipred_paeth_16bpc,      avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_16bpc,     avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_h_16bpc,   avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_v_16bpc,   avx2, w4, w8, w16, w32, w64

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

%macro PAETH 3 ; top, signed_ldiff, ldiff
    paddw               m0, m%2, m1
    psubw               m7, m3, m0  ; tldiff
    psubw               m0, m%1     ; tdiff
    pabsw               m7, m7
    pabsw               m0, m0
    pminsw              m7, m0
    pcmpeqw             m0, m7
    pcmpgtw             m7, m%3, m7
    vpblendvb           m0, m3, m%1, m0
    vpblendvb           m0, m1, m0, m7
%endmacro

cglobal ipred_paeth_16bpc, 3, 6, 8, dst, stride, tl, w, h
%define base r5-ipred_paeth_16bpc_avx2_table
    movifnidn           hd, hm
    lea                 r5, [ipred_paeth_16bpc_avx2_table]
    tzcnt               wd, wd
    movsxd              wq, [r5+wq*4]
    vpbroadcastw        m3, [tlq]   ; topleft
    add                 wq, r5
    jmp                 wq
.w4:
    vpbroadcastq        m2, [tlq+2] ; top
    movsldup            m6, [base+ipred_hv_shuf]
    lea                 r3, [strideq*3]
    psubw               m4, m2, m3
    pabsw               m5, m4
.w4_loop:
    sub                tlq, 8
    vpbroadcastq        m1, [tlq]
    pshufb              m1, m6      ; left
    PAETH                2, 4, 5
    vextracti128       xm1, m0, 1
    movq  [dstq+strideq*0], xm0
    movq  [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea               dstq, [dstq+strideq*4]
    sub                 hd, 4
    jg .w4_loop
    RET
ALIGN function_align
.w8:
    vbroadcasti128      m2, [tlq+2]
    movsldup            m6, [base+ipred_hv_shuf]
    lea                 r3, [strideq*3]
    psubw               m4, m2, m3
    pabsw               m5, m4
.w8_loop:
    sub                tlq, 4
    vpbroadcastd        m1, [tlq]
    pshufb              m1, m6
    PAETH                2, 4, 5
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea               dstq, [dstq+strideq*2]
    sub                 hd, 2
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    movu                m2, [tlq+2]
    psubw               m4, m2, m3
    pabsw               m5, m4
.w16_loop:
    sub                tlq, 2
    vpbroadcastw        m1, [tlq]
    PAETH                2, 4, 5
    mova            [dstq], m0
    add               dstq, strideq
    dec                 hd
    jg .w16_loop
    RET
ALIGN function_align
.w32:
    movu                m2, [tlq+2]
    movu                m6, [tlq+34]
%if WIN64
    movaps             r4m, xmm8
    movaps             r6m, xmm9
%endif
    psubw               m4, m2, m3
    psubw               m8, m6, m3
    pabsw               m5, m4
    pabsw               m9, m8
.w32_loop:
    sub                tlq, 2
    vpbroadcastw        m1, [tlq]
    PAETH                2, 4, 5
    mova       [dstq+32*0], m0
    PAETH                6, 8, 9
    mova       [dstq+32*1], m0
    add               dstq, strideq
    dec                 hd
    jg .w32_loop
%if WIN64
    movaps            xmm8, r4m
    movaps            xmm9, r6m
%endif
    RET
ALIGN function_align
.w64:
    WIN64_SPILL_XMM 16
    movu                m2, [tlq+ 2]
    movu                m6, [tlq+34]
    movu               m10, [tlq+66]
    movu               m13, [tlq+98]
    psubw               m4, m2, m3
    psubw               m8, m6, m3
    psubw              m11, m10, m3
    psubw              m14, m13, m3
    pabsw               m5, m4
    pabsw               m9, m8
    pabsw              m12, m11
    pabsw              m15, m14
.w64_loop:
    sub                tlq, 2
    vpbroadcastw        m1, [tlq]
    PAETH                2, 4, 5
    mova       [dstq+32*0], m0
    PAETH                6, 8, 9
    mova       [dstq+32*1], m0
    PAETH               10, 11, 12
    mova       [dstq+32*2], m0
    PAETH               13, 14, 15
    mova       [dstq+32*3], m0
    add               dstq, strideq
    dec                 hd
    jg .w64_loop
    RET

%macro SMOOTH 4 ; src[1-2], mul[1-2]
    pmaddwd              m0, m%3, m%1
    pmaddwd              m1, m%4, m%2
    paddd                m0, m2
    paddd                m1, m2
    psrld                m0, 8
    psrld                m1, 8
    packssdw             m0, m1
%endmacro

cglobal ipred_smooth_v_16bpc, 3, 7, 6, dst, stride, tl, w, h, weights
%define base r6-ipred_smooth_v_16bpc_avx2_table
    lea                 r6, [ipred_smooth_v_16bpc_avx2_table]
    tzcnt               wd, wm
    mov                 hd, hm
    movsxd              wq, [r6+wq*4]
    vpbroadcastd        m2, [base+pd_128]
    lea            weightsq, [base+smooth_weights+hq*8]
    neg                 hq
    vpbroadcastw        m5, [tlq+hq*2] ; bottom
    add                 wq, r6
    jmp                 wq
.w4:
    vpbroadcastq        m3, [tlq+2]
    punpcklwd           m3, m5 ; top, bottom
    movshdup            m5, [base+ipred_hv_shuf]
    lea                 r3, [strideq*3]
    punpcklqdq          m4, m5, m5
    punpckhqdq          m5, m5
.w4_loop:
    vbroadcasti128      m1, [weightsq+hq*4]
    pshufb              m0, m1, m4
    pshufb              m1, m5
    SMOOTH               3, 3, 0, 1
    vextracti128       xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea               dstq, [dstq+strideq*4]
    add                 hq, 4
    jl .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    vbroadcasti128      m4, [tlq+2]
    punpcklwd           m3, m4, m5
    punpckhwd           m4, m5
    movshdup            m5, [base+ipred_hv_shuf]
.w8_loop:
    vpbroadcastq        m1, [weightsq+hq*4]
    pshufb              m1, m5
    SMOOTH               3, 4, 1, 1
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea               dstq, [dstq+strideq*2]
    add                 hq, 2
    jl .w8_loop
    RET
ALIGN function_align
.w16:
    movu                m4, [tlq+2]
    punpcklwd           m3, m4, m5
    punpckhwd           m4, m5
.w16_loop:
    vpbroadcastd        m1, [weightsq+hq*4]
    vpbroadcastd        m5, [weightsq+hq*4+4]
    SMOOTH               3, 4, 1, 1
    mova  [dstq+strideq*0], m0
    SMOOTH               3, 4, 5, 5
    mova  [dstq+strideq*1], m0
    lea               dstq, [dstq+strideq*2]
    add                 hq, 2
    jl .w16_loop
    RET
ALIGN function_align
.w32:
    WIN64_SPILL_XMM      8
    movu                m4, [tlq+2]
    movu                m7, [tlq+34]
    punpcklwd           m3, m4, m5
    punpckhwd           m4, m5
    punpcklwd           m6, m7, m5
    punpckhwd           m7, m5
.w32_loop:
    vpbroadcastd        m5, [weightsq+hq*4]
    SMOOTH               3, 4, 5, 5
    mova       [dstq+32*0], m0
    SMOOTH               6, 7, 5, 5
    mova       [dstq+32*1], m0
    add               dstq, strideq
    inc                 hq
    jl .w32_loop
    RET
ALIGN function_align
.w64:
    WIN64_SPILL_XMM     12
    movu                m4, [tlq+ 2]
    movu                m7, [tlq+34]
    movu                m9, [tlq+66]
    movu               m11, [tlq+98]
    punpcklwd           m3, m4, m5
    punpckhwd           m4, m5
    punpcklwd           m6, m7, m5
    punpckhwd           m7, m5
    punpcklwd           m8, m9, m5
    punpckhwd           m9, m5
    punpcklwd          m10, m11, m5
    punpckhwd          m11, m5
.w64_loop:
    vpbroadcastd        m5, [weightsq+hq*4]
    SMOOTH               3, 4, 5, 5
    mova       [dstq+32*0], m0
    SMOOTH               6, 7, 5, 5
    mova       [dstq+32*1], m0
    SMOOTH               8, 9, 5, 5
    mova       [dstq+32*2], m0
    SMOOTH              10, 11, 5, 5
    mova       [dstq+32*3], m0
    add               dstq, strideq
    inc                 hq
    jl .w64_loop
    RET

cglobal ipred_smooth_h_16bpc, 3, 7, 6, dst, stride, tl, w, h
%define base r6-ipred_smooth_h_16bpc_avx2_table
    lea                 r6, [ipred_smooth_h_16bpc_avx2_table]
    mov                 wd, wm
    mov                 hd, hm
    vpbroadcastw        m3, [tlq+wq*2] ; right
    tzcnt               wd, wd
    movsxd              wq, [r6+wq*4]
    vpbroadcastd        m2, [base+pd_128]
    add                 wq, r6
    jmp                 wq
.w4:
    vbroadcasti128      m4, [base+smooth_weights+4*4]
    movsldup            m5, [base+ipred_hv_shuf]
    sub                tlq, 8
    sub                tlq, hq
    sub                tlq, hq
    lea                 r3, [strideq*3]
.w4_loop:
    vpbroadcastq        m1, [tlq+hq*2]
    pshufb              m1, m5
    punpcklwd           m0, m1, m3 ; left, right
    punpckhwd           m1, m3
    SMOOTH               0, 1, 4, 4
    vextracti128       xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea               dstq, [dstq+strideq*4]
    sub                 hd, 4
    jg .w4_loop
    RET
ALIGN function_align
.w8:
    WIN64_SPILL_XMM      7
    vbroadcasti128      m4, [base+smooth_weights+8*4+16*0]
    vbroadcasti128      m5, [base+smooth_weights+8*4+16*1]
    movsldup            m6, [base+ipred_hv_shuf]
    sub                tlq, 4
    sub                tlq, hq
    sub                tlq, hq
.w8_loop:
    vpbroadcastd        m1, [tlq+hq*2]
    pshufb              m1, m6
    punpcklwd           m0, m1, m3
    punpckhwd           m1, m3
    SMOOTH               0, 1, 4, 5
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea               dstq, [dstq+strideq*2]
    sub                 hq, 2
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    WIN64_SPILL_XMM      6
    mova               xm4, [base+smooth_weights+16*4+16*0]
    mova               xm5, [base+smooth_weights+16*4+16*1]
    vinserti128         m4, [base+smooth_weights+16*4+16*2], 1
    vinserti128         m5, [base+smooth_weights+16*4+16*3], 1
    sub                tlq, 2
    sub                tlq, hq
    sub                tlq, hq
.w16_loop:
    vpbroadcastw        m1, [tlq+hq*2]
    punpcklwd           m0, m1, m3
    punpckhwd           m1, m3
    SMOOTH               0, 1, 4, 5
    mova            [dstq], m0
    add               dstq, strideq
    dec                 hq
    jg .w16_loop
    RET
ALIGN function_align
.w32:
    WIN64_SPILL_XMM     10
    mova               xm6, [base+smooth_weights+32*4+16*0]
    mova               xm7, [base+smooth_weights+32*4+16*1]
    vinserti128         m6, [base+smooth_weights+32*4+16*2], 1
    vinserti128         m7, [base+smooth_weights+32*4+16*3], 1
    mova               xm8, [base+smooth_weights+32*4+16*4]
    mova               xm9, [base+smooth_weights+32*4+16*5]
    vinserti128         m8, [base+smooth_weights+32*4+16*6], 1
    vinserti128         m9, [base+smooth_weights+32*4+16*7], 1
    sub                tlq, 2
    sub                tlq, hq
    sub                tlq, hq
.w32_loop:
    vpbroadcastw        m5, [tlq+hq*2]
    punpcklwd           m4, m5, m3
    punpckhwd           m5, m3
    SMOOTH               4, 5, 6, 7
    mova       [dstq+32*0], m0
    SMOOTH               4, 5, 8, 9
    mova       [dstq+32*1], m0
    add               dstq, strideq
    dec                 hq
    jg .w32_loop
    RET
ALIGN function_align
.w64:
%assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM     14
    mova               xm6, [base+smooth_weights+64*4+16* 0]
    mova               xm7, [base+smooth_weights+64*4+16* 1]
    vinserti128         m6, [base+smooth_weights+64*4+16* 2], 1
    vinserti128         m7, [base+smooth_weights+64*4+16* 3], 1
    mova               xm8, [base+smooth_weights+64*4+16* 4]
    mova               xm9, [base+smooth_weights+64*4+16* 5]
    vinserti128         m8, [base+smooth_weights+64*4+16* 6], 1
    vinserti128         m9, [base+smooth_weights+64*4+16* 7], 1
    mova              xm10, [base+smooth_weights+64*4+16* 8]
    mova              xm11, [base+smooth_weights+64*4+16* 9]
    vinserti128        m10, [base+smooth_weights+64*4+16*10], 1
    vinserti128        m11, [base+smooth_weights+64*4+16*11], 1
    mova              xm12, [base+smooth_weights+64*4+16*12]
    mova              xm13, [base+smooth_weights+64*4+16*13]
    vinserti128        m12, [base+smooth_weights+64*4+16*14], 1
    vinserti128        m13, [base+smooth_weights+64*4+16*15], 1
    sub                tlq, 2
    sub                tlq, hq
    sub                tlq, hq
.w64_loop:
    vpbroadcastw        m5, [tlq+hq*2]
    punpcklwd           m4, m5, m3
    punpckhwd           m5, m3
    SMOOTH               4, 5, 6, 7
    mova       [dstq+32*0], m0
    SMOOTH               4, 5, 8, 9
    mova       [dstq+32*1], m0
    SMOOTH               4, 5, 10, 11
    mova       [dstq+32*2], m0
    SMOOTH               4, 5, 12, 13
    mova       [dstq+32*3], m0
    add               dstq, strideq
    dec                 hq
    jg .w64_loop
    RET

%macro SMOOTH_2D_END 6 ; src[1-2], mul[1-2], add[1-2]
    pmaddwd             m0, m%1, m%3
    pmaddwd             m1, m%2, m%4
    paddd               m0, m%5
    paddd               m1, m%6
    paddd               m0, m5
    paddd               m1, m5
    psrld               m0, 9
    psrld               m1, 9
    packssdw            m0, m1
%endmacro

cglobal ipred_smooth_16bpc, 3, 7, 6, dst, stride, tl, w, h, v_weights
%define base r6-ipred_smooth_16bpc_avx2_table
    lea                 r6, [ipred_smooth_16bpc_avx2_table]
    mov                 wd, wm
    vpbroadcastw        m4, [tlq+wq*2] ; right
    tzcnt               wd, wd
    mov                 hd, hm
    sub                tlq, hq
    sub                tlq, hq
    movsxd              wq, [r6+wq*4]
    vpbroadcastd        m5, [base+pd_256]
    add                 wq, r6
    lea         v_weightsq, [base+smooth_weights+hq*4]
    jmp                 wq
.w4:
    WIN64_SPILL_XMM     11
    vpbroadcastw        m0, [tlq] ; bottom
    vpbroadcastq        m6, [tlq+hq*2+2]
    movsldup            m7, [base+ipred_hv_shuf]
    movshdup            m9, [base+ipred_hv_shuf]
    vbroadcasti128     m10, [base+smooth_weights+4*4]
    punpcklwd           m6, m0 ; top, bottom
    punpcklqdq          m8, m9, m9
    punpckhqdq          m9, m9
    lea                 r3, [strideq*3]
    sub                tlq, 8
.w4_loop:
    vbroadcasti128      m1, [v_weightsq]
    vpbroadcastq        m3, [tlq+hq*2]
    pshufb              m3, m7
    punpcklwd           m2, m3, m4 ; left, right
    punpckhwd           m3, m4
    pmaddwd             m2, m10
    pmaddwd             m3, m10
    pshufb              m0, m1, m8
    pshufb              m1, m9
    SMOOTH_2D_END        6, 6, 0, 1, 2, 3
    vextracti128       xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea               dstq, [dstq+strideq*4]
    add         v_weightsq, 16
    sub                 hd, 4
    jg .w4_loop
    RET
ALIGN function_align
.w8:
%assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM     12
    vpbroadcastw        m0, [tlq] ; bottom
    vbroadcasti128      m7, [tlq+hq*2+2]
    movsldup            m8, [base+ipred_hv_shuf]
    movshdup            m9, [base+ipred_hv_shuf]
    vbroadcasti128     m10, [base+smooth_weights+8*4+16*0]
    vbroadcasti128     m11, [base+smooth_weights+8*4+16*1]
    punpcklwd           m6, m7, m0 ; top, bottom
    punpckhwd           m7, m0
    sub                tlq, 4
.w8_loop:
    vpbroadcastq        m1, [v_weightsq]
    vpbroadcastd        m3, [tlq+hq*2]
    pshufb              m3, m8
    punpcklwd           m2, m3, m4 ; left, right
    punpckhwd           m3, m4
    pmaddwd             m2, m10
    pmaddwd             m3, m11
    pshufb              m1, m9
    SMOOTH_2D_END        6, 7, 1, 1, 2, 3
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea               dstq, [dstq+strideq*2]
    add         v_weightsq, 8
    sub                 hd, 2
    jg .w8_loop
    RET
ALIGN function_align
.w16:
%assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM     14
    vpbroadcastw        m0, [tlq] ; bottom
    movu                m7, [tlq+hq*2+2]
    mova               xm8, [base+smooth_weights+16*4+16*0]
    mova               xm9, [base+smooth_weights+16*4+16*1]
    vinserti128         m8, [base+smooth_weights+16*4+16*2], 1
    vinserti128         m9, [base+smooth_weights+16*4+16*3], 1
    punpcklwd           m6, m7, m0 ; top, bottom
    punpckhwd           m7, m0
    sub                tlq, 2
.w16_loop:
    vpbroadcastd       m10, [v_weightsq+0]
    vpbroadcastd       m11, [v_weightsq+4]
    vpbroadcastw        m3, [tlq+hq*2-0]
    vpbroadcastw       m13, [tlq+hq*2-2]
    punpcklwd           m2, m3, m4 ; left, right
    punpckhwd           m3, m4
    punpcklwd          m12, m13, m4
    punpckhwd          m13, m4
    pmaddwd             m2, m8
    pmaddwd             m3, m9
    pmaddwd            m12, m8
    pmaddwd            m13, m9
    SMOOTH_2D_END        6, 7, 10, 10, 2, 3
    mova  [dstq+strideq*0], m0
    SMOOTH_2D_END        6, 7, 11, 11, 12, 13
    mova  [dstq+strideq*1], m0
    lea               dstq, [dstq+strideq*2]
    add         v_weightsq, 8
    sub                 hq, 2
    jg .w16_loop
    RET
ALIGN function_align
.w32:
%assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM     16
    vpbroadcastw        m0, [tlq] ; bottom
    movu                m7, [tlq+hq*2+ 2]
    movu                m9, [tlq+hq*2+34]
    mova              xm10, [base+smooth_weights+32*4+16*0]
    mova              xm11, [base+smooth_weights+32*4+16*1]
    vinserti128        m10, [base+smooth_weights+32*4+16*2], 1
    vinserti128        m11, [base+smooth_weights+32*4+16*3], 1
    mova              xm12, [base+smooth_weights+32*4+16*4]
    mova              xm13, [base+smooth_weights+32*4+16*5]
    vinserti128        m12, [base+smooth_weights+32*4+16*6], 1
    vinserti128        m13, [base+smooth_weights+32*4+16*7], 1
    punpcklwd           m6, m7, m0
    punpckhwd           m7, m0
    punpcklwd           m8, m9, m0
    punpckhwd           m9, m0
    sub                tlq, 2
.w32_loop:
    vpbroadcastw        m3, [tlq+hq*2]
    punpcklwd           m2, m3, m4
    punpckhwd           m3, m4
    pmaddwd            m14, m2, m10
    pmaddwd            m15, m3, m11
    pmaddwd             m2, m12
    pmaddwd             m3, m13
    vpbroadcastd        m1, [v_weightsq]
    pmaddwd             m0, m6, m1
    paddd               m0, m14
    paddd               m0, m5
    psrld               m0, 9
    pmaddwd            m14, m7, m1
    paddd              m14, m15
    paddd              m14, m5
    psrld              m14, 9
    packssdw            m0, m14
    mova       [dstq+32*0], m0
    SMOOTH_2D_END        8, 9, 1, 1, 2, 3
    mova       [dstq+32*1], m0
    add               dstq, strideq
    add         v_weightsq, 4
    dec                 hd
    jg .w32_loop
    RET
ALIGN function_align
.w64:
%assign stack_offset stack_offset - stack_size_padded
    PROLOGUE 0, 11, 16, dst, stride, tl, tl_base, h, v_weights, dummy, v_weights_base, x, y, dst_base
    mov          dst_baseq, dstq
    mov           tl_baseq, tlq
    mov    v_weights_baseq, v_weightsq
    xor                 xq, xq
.w64_loop_x:
    mov                 yq, hq
    lea                tlq, [tl_baseq+hq*2]
    vpbroadcastw        m0, [tl_baseq] ; bottom
    movu                m7, [tlq+xq*2+ 2]
    movu                m9, [tlq+xq*2+34]
    mova              xm10, [base+smooth_weights+64*4+16*0]
    mova              xm11, [base+smooth_weights+64*4+16*1]
    vinserti128        m10, [base+smooth_weights+64*4+16*2], 1
    vinserti128        m11, [base+smooth_weights+64*4+16*3], 1
    mova              xm12, [base+smooth_weights+64*4+16*4]
    mova              xm13, [base+smooth_weights+64*4+16*5]
    vinserti128        m12, [base+smooth_weights+64*4+16*6], 1
    vinserti128        m13, [base+smooth_weights+64*4+16*7], 1
    punpcklwd           m6, m7, m0
    punpckhwd           m7, m0
    punpcklwd           m8, m9, m0
    punpckhwd           m9, m0
    lea                tlq, [tl_baseq-2]
.w64_loop_y:
    vpbroadcastd        m1, [v_weightsq]
    vpbroadcastw        m3, [tlq+yq*2]
    punpcklwd           m2, m3, m4
    punpckhwd           m3, m4
    pmaddwd            m14, m2, m10
    pmaddwd            m15, m3, m11
    pmaddwd             m2, m12
    pmaddwd             m3, m13
    pmaddwd             m0, m6, m1
    paddd               m0, m14
    paddd               m0, m5
    psrld               m0, 9
    pmaddwd            m14, m7, m1
    paddd              m14, m15
    paddd              m14, m5
    psrld              m14, 9
    packssdw            m0, m14
    mova       [dstq+32*0], m0
    SMOOTH_2D_END        8, 9, 1, 1, 2, 3
    mova       [dstq+32*1], m0
    add               dstq, strideq
    add         v_weightsq, 4
    dec                 yq
    jg .w64_loop_y
    lea               dstq, [dst_baseq+32*2]
    add                 r6, 16*8
    mov         v_weightsq, v_weights_baseq
    add                 xq, 32
    test                xb, 64
    jz .w64_loop_x
    RET

%endif
