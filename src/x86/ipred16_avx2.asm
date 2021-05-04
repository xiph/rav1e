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
filter_shuf1:  db  8,  9,  0,  1,  2,  3,  4,  5,  6,  7, 14, 15, 12, 13, -1, -1
filter_shuf2:  db  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  4,  5,  2,  3, -1, -1
filter_shuf3:  db 12, 13,  0,  1,  2,  3,  4,  5,  6,  7, 10, 11,  8,  9, -1, -1
pal_pred_shuf: db  0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15
z_base_inc:    dw   0*64,   1*64,   2*64,   3*64,   4*64,   5*64,   6*64,   7*64
               dw   8*64,   9*64,  10*64,  11*64,  12*64,  13*64,  14*64,  15*64
z_filter_t0:   db 55,127, 39,127, 39,127,  7, 15, 31,  7, 15, 31,  0,  3, 31,  0
z_filter_t1:   db 39, 63, 19, 47, 19, 47,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0
z_filter_wh:   db  7,  7, 11, 11, 15, 15, 19, 19, 19, 23, 23, 23, 31, 31, 31, 39
               db 39, 39, 47, 47, 47, 63, 63, 63, 79, 79, 79, -1
pw_m1024:      times 2 dw -1024
z_upsample:    db  0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15
z_filter_k:    dw  4,  4,  5,  5,  4,  4,  8,  8,  6,  6,  4,  4

%define pw_4 z_filter_k

pw_2:    times 2 dw 2
pw_3:    times 2 dw 3
pw_62:   times 2 dw 62
pw_512:  times 2 dw 512
pw_2048: times 2 dw 2048
pd_8:    dd 8
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
%define ipred_cfl_splat_16bpc_avx2_table (ipred_cfl_16bpc_avx2_table + 8*4)

JMP_TABLE ipred_dc_16bpc,         avx2, h4, h8, h16, h32, h64, w4, w8, w16, w32, w64, \
                                        s4-10*4, s8-10*4, s16-10*4, s32-10*4, s64-10*4
JMP_TABLE ipred_dc_left_16bpc,    avx2, h4, h8, h16, h32, h64
JMP_TABLE ipred_h_16bpc,          avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_paeth_16bpc,      avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_16bpc,     avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_h_16bpc,   avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_v_16bpc,   avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_z1_16bpc,         avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_filter_16bpc,     avx2, w4, w8, w16, w32
JMP_TABLE ipred_cfl_16bpc,        avx2, h4, h8, h16, h32, w4, w8, w16, w32, \
                                        s4-8*4, s8-8*4, s16-8*4, s32-8*4
JMP_TABLE ipred_cfl_left_16bpc,   avx2, h4, h8, h16, h32
JMP_TABLE ipred_cfl_ac_420_16bpc, avx2, w16_wpad_pad1, w16_wpad_pad2, w16_wpad_pad3
JMP_TABLE ipred_cfl_ac_422_16bpc, avx2, w16_wpad_pad1, w16_wpad_pad2, w16_wpad_pad3
JMP_TABLE pal_pred_16bpc,         avx2, w4, w8, w16, w32, w64

cextern dr_intra_derivative
cextern filter_intra_taps

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

cglobal ipred_z1_16bpc, 3, 8, 0, dst, stride, tl, w, h, angle, dx, maxbase
    %assign org_stack_offset stack_offset
    lea                  r6, [ipred_z1_16bpc_avx2_table]
    tzcnt                wd, wm
    movifnidn        angled, anglem
    movifnidn            hd, hm
    lea                  r7, [dr_intra_derivative]
    movsxd               wq, [r6+wq*4]
    add                 tlq, 2
    add                  wq, r6
    mov                 dxd, angled
    and                 dxd, 0x7e
    add              angled, 165 ; ~90
    movzx               dxd, word [r7+dxq]
    xor              angled, 0x4ff ; d = 90 - angle
    vpbroadcastd         m5, [pw_62]
    jmp                  wq
.w4:
    ALLOC_STACK         -64, 7
    cmp              angleb, 40
    jae .w4_no_upsample
    lea                 r3d, [angleq-1024]
    sar                 r3d, 7
    add                 r3d, hd
    jg .w4_no_upsample ; !enable_intra_edge_filter || h > 8 || (h == 8 && is_sm)
    vpbroadcastw        xm3, [tlq+14]
    movu                xm1, [tlq+ 0]    ; 1 2 3 4 5 6 7 8
    palignr             xm0, xm3, xm1, 4 ; 3 4 5 6 7 8 8 8
    paddw               xm0, [tlq- 2]    ; 0 1 2 3 4 5 6 7
    add                 dxd, dxd
    palignr             xm2, xm3, xm1, 2 ; 2 3 4 5 6 7 8 8
    paddw               xm2, xm1         ; -1 * a + 9 * b + 9 * c + -1 * d
    psubw               xm0, xm2, xm0    ; = (b + c - a - d + (b + c) << 3 + 8) >> 4
    psraw               xm0, 3           ; = ((b + c - a - d) >> 3 + b + c + 1) >> 1
    pxor                xm4, xm4
    paddw               xm2, xm0
    vpbroadcastw        xm0, r8m         ; pixel_max
    mova           [rsp+32], xm3
    movd                xm3, dxd
    pmaxsw              xm2, xm4
    mov                 r3d, dxd
    pavgw               xm2, xm4
    vpbroadcastw         m3, xm3
    pminsw              xm2, xm0
    punpcklwd           xm0, xm1, xm2
    punpckhwd           xm1, xm2
    lea                  r5, [strideq*3]
    pslldq               m2, m3, 8
    mova           [rsp+ 0], xm0
    mova           [rsp+16], xm1
    paddw                m6, m3, m3
    paddw                m3, m2
    vpblendd             m4, m6, 0xf0
    paddw                m6, m6
    paddw                m3, m4 ; xpos0 xpos1 xpos2 xpos3
    vbroadcasti128       m4, [z_upsample]
.w4_upsample_loop:
    lea                 r2d, [r3+dxq]
    shr                 r3d, 6 ; base0
    movu                xm1, [rsp+r3*2]
    lea                 r3d, [r2+dxq]
    shr                 r2d, 6 ; base1
    movu                xm2, [rsp+r2*2]
    lea                 r2d, [r3+dxq]
    shr                 r3d, 6 ; base2
    vinserti128          m1, [rsp+r3*2], 1 ; 0 2
    lea                 r3d, [r2+dxq]
    shr                 r2d, 6 ; base3
    vinserti128          m2, [rsp+r2*2], 1 ; 1 3
    pshufb               m1, m4
    pshufb               m2, m4
    punpcklqdq           m0, m1, m2
    punpckhqdq           m1, m2
    pand                 m2, m5, m3 ; frac
    psllw                m2, 9      ; (a * (64 - frac) + b * frac + 32) >> 6
    psubw                m1, m0     ; = a + (((b - a) * frac + 32) >> 6)
    pmulhrsw             m1, m2     ; = a + (((b - a) * (frac << 9) + 16384) >> 15)
    paddw                m3, m6     ; xpos += dx
    paddw                m0, m1
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movhps [dstq+strideq*1], xm0
    movq   [dstq+strideq*2], xm1
    movhps [dstq+r5       ], xm1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w4_upsample_loop
    RET
ALIGN function_align
.filter_strength: ; w4/w8/w16
%define base r3-z_filter_t0
    movd                xm0, maxbased
    lea                  r3, [z_filter_t0]
    movd                xm1, angled
    shr              angled, 8 ; is_sm << 1
    vpbroadcastb         m0, xm0
    vpbroadcastb         m1, xm1
    pcmpeqb              m0, [base+z_filter_wh]
    mova                xm2, [r3+angleq*8]
    pand                 m0, m1
    pcmpgtb              m0, m2
    pmovmskb            r5d, m0
    ret
.w4_no_upsample:
    mov            maxbased, 7
    test             angled, 0x400 ; !enable_intra_edge_filter
    jnz .w4_main
    lea            maxbased, [hq+3]
    call .filter_strength
    mov            maxbased, 7
    test                r5d, r5d
    jz .w4_main ; filter_strength == 0
    popcnt              r5d, r5d
    vpbroadcastw        xm3, [tlq+14]
    mova                xm0, [tlq- 2]      ; 0 1 2 3 4 5 6 7
    vpbroadcastd        xm1, [base+z_filter_k-4+r5*4+12*1]
    vpbroadcastd        xm4, [base+z_filter_k-4+r5*4+12*0]
    palignr             xm2, xm3, xm0, 4   ; 2 3 4 5 6 7 8 8
    pmullw              xm1, [tlq+ 0]      ; 1 2 3 4 5 6 7 8
    paddw               xm2, xm0
    pmullw              xm2, xm4
    movd           [rsp+16], xm3
    cmp                 r5d, 3
    jne .w4_3tap
    paddw               xm1, xm2
    palignr             xm2, xm3, xm0, 6   ; 3 4 5 6 7 8 8 8
    pblendw             xm0, [tlq-4], 0xfe ; 0 0 1 2 3 4 5 6
    movzx               r3d, word [tlq+14]
    movzx               r2d, word [tlq+12]
    inc            maxbased
    paddw               xm2, xm0
    sub                 r2d, r3d
    paddw               xm2, xm2
    lea                 r2d, [r2+r3*8+4]
    shr                 r2d, 3 ; (1 * top[6] + 7 * top[7] + 4) >> 3
    mov            [rsp+16], r2w
.w4_3tap:
    pxor                xm0, xm0
    paddw               xm1, xm2
    mov                 tlq, rsp
    psrlw               xm1, 3
    cmp                  hd, 8
    sbb            maxbased, -1
    pavgw               xm0, xm1
    mova              [tlq], xm0
.w4_main:
    movd                xm3, dxd
    vpbroadcastq         m1, [z_base_inc]
    vpbroadcastw         m6, [tlq+maxbaseq*2] ; top[max_base_x]
    shl            maxbased, 6
    vpbroadcastw         m3, xm3
    movd                xm0, maxbased
    mov                 r3d, dxd      ; xpos
    vpbroadcastw         m0, xm0
    paddw                m4, m3, m3
    psubw                m1, m0       ; -max_base_x
    vpblendd             m3, m4, 0xcc
    paddw                m0, m4, m3
    vpblendd             m3, m0, 0xf0 ; xpos0 xpos1 xpos2 xpos3
    paddw                m4, m4
    paddw                m3, m1
.w4_loop:
    lea                 r5d, [r3+dxq]
    shr                 r3d, 6 ; base0
    movu                xm1, [tlq+r3*2]
    lea                 r3d, [r5+dxq]
    shr                 r5d, 6 ; base1
    movu                xm2, [tlq+r5*2]
    lea                 r5d, [r3+dxq]
    shr                 r3d, 6 ; base2
    vinserti128          m1, [tlq+r3*2], 1 ; 0 2
    lea                 r3d, [r5+dxq]
    shr                 r5d, 6 ; base3
    vinserti128          m2, [tlq+r5*2], 1 ; 1 3
    punpcklqdq           m0, m1, m2
    psrldq               m1, 2
    pslldq               m2, 6
    vpblendd             m1, m2, 0xcc
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    psraw                m2, m3, 15 ; xpos < max_base_x
    paddw                m3, m4
    paddw                m0, m1
    vpblendvb            m0, m6, m0, m2
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movhps [dstq+strideq*1], xm0
    lea                dstq, [dstq+strideq*2]
    movq   [dstq+strideq*0], xm1
    movhps [dstq+strideq*1], xm1
    sub                  hd, 4
    jz .w4_end
    lea                dstq, [dstq+strideq*2]
    cmp                 r3d, maxbased
    jb .w4_loop
    lea                  r6, [strideq*3]
.w4_end_loop:
    movq   [dstq+strideq*0], xm6
    movq   [dstq+strideq*1], xm6
    movq   [dstq+strideq*2], xm6
    movq   [dstq+r6       ], xm6
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w4_end_loop
.w4_end:
    RET
.w8:
    %assign stack_offset org_stack_offset
    ALLOC_STACK         -64, 7
    lea                 r3d, [angleq+216]
    mov                 r3b, hb
    cmp                 r3d, 8
    ja .w8_no_upsample ; !enable_intra_edge_filter || is_sm || d >= 40 || h > 8
    movu                 m2, [tlq+2]    ; 2 3 4 5 6 7 8 9   a b c d e f g _
    movu                 m0, [tlq+4]    ; 3 4 5 6 7 8 9 a   b c d e f g _ _
    movu                 m1, [tlq+0]    ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    cmp                  hd, 4
    jne .w8_upsample_h8 ; awkward single-pixel edge case
    vpblendd             m0, m2, 0x20   ; 3 4 5 6 7 8 9 a   b c c _ _ _ _ _
.w8_upsample_h8:
    paddw                m2, m1
    paddw                m0, [tlq-2]    ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    add                 dxd, dxd
    psubw                m0, m2, m0
    psraw                m0, 3
    pxor                 m4, m4
    paddw                m2, m0
    vpbroadcastw         m0, r8m
    movd                xm3, dxd
    pmaxsw               m2, m4
    mov                 r3d, dxd
    pavgw                m2, m4
    vpbroadcastw         m3, xm3
    pminsw               m2, m0
    punpcklwd            m0, m1, m2
    punpckhwd            m1, m2
    vbroadcasti128       m4, [z_upsample]
    mova           [rsp+ 0], xm0
    mova           [rsp+16], xm1
    paddw                m6, m3, m3
    vextracti128   [rsp+32], m0, 1
    vextracti128   [rsp+48], m1, 1
    vpblendd             m3, m6, 0xf0 ; xpos0 xpos1
.w8_upsample_loop:
    lea                 r2d, [r3+dxq]
    shr                 r3d, 6 ; base0
    movu                xm1, [rsp+r3*2]
    movu                xm2, [rsp+r3*2+16]
    lea                 r3d, [r2+dxq]
    shr                 r2d, 6 ; base1
    vinserti128          m1, [rsp+r2*2], 1
    vinserti128          m2, [rsp+r2*2+16], 1
    pshufb               m1, m4
    pshufb               m2, m4
    punpcklqdq           m0, m1, m2
    punpckhqdq           m1, m2
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    paddw                m3, m6
    paddw                m0, m1
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w8_upsample_loop
    RET
.w8_no_intra_edge_filter:
    and            maxbased, 7
    or             maxbased, 8 ; imin(h+7, 15)
    jmp .w8_main
.w8_no_upsample:
    lea            maxbased, [hq+7]
    test             angled, 0x400
    jnz .w8_no_intra_edge_filter
    call .filter_strength
    test                r5d, r5d
    jz .w8_main
    popcnt              r5d, r5d
    vpbroadcastd         m1, [base+z_filter_k-4+r5*4+12*1]
    vpbroadcastd         m4, [base+z_filter_k-4+r5*4+12*0]
    mova                 m0, [tlq-2]           ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    movu                 m2, [tlq+0]           ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    pmullw               m1, m2
    cmp                  hd, 8
    jl .w8_filter_h4
    punpckhwd            m2, m2
    vpblendd             m3, m2, [tlq+2], 0x7f ; 2 3 4 5 6 7 8 9   a b c d e f g g
    je .w8_filter_end ; 8x4 and 8x8 are always 3-tap
    movzx               r3d, word [tlq+30]
    mov            maxbased, 16
    mov            [rsp+32], r3d
    cmp                 r5d, 3
    jne .w8_filter_end
    punpcklwd           xm6, xm0, xm0
    vpblendd             m2, [tlq+4], 0x7f     ; 3 4 5 6 7 8 9 a   b c d e f g g g
    vpblendd             m6, [tlq-4], 0xfe     ; 0 0 1 2 3 4 5 6   7 8 9 a b c d e
    movzx               r5d, word [tlq+28]
    mov            [rsp+34], r3w
    paddw                m2, m6
    sub                 r5d, r3d
    inc            maxbased
    paddw                m2, m2
    lea                 r3d, [r5+r3*8+4]
    paddw                m1, m2
    shr                 r3d, 3
    mov            [rsp+32], r3w
    jmp .w8_filter_end
.w8_filter_h4:
    pshuflw              m3, m2, q3321
    vinserti128          m3, [tlq+2], 0        ; 2 3 4 5 6 7 8 9   a b c c _ _ _ _
.w8_filter_end:
    paddw                m0, m3
    pmullw               m0, m4
    mov                 tlq, rsp
    pxor                 m2, m2
    paddw                m0, m1
    psrlw                m0, 3
    pavgw                m0, m2
    mova              [tlq], m0
.w8_main:
    movd                xm3, dxd
    vbroadcasti128       m1, [z_base_inc]
    vpbroadcastw         m6, [tlq+maxbaseq*2]
    shl            maxbased, 6
    vpbroadcastw         m3, xm3
    movd                xm0, maxbased
    mov                 r3d, dxd
    vpbroadcastw         m0, xm0
    paddw                m4, m3, m3
    psubw                m1, m0
    vpblendd             m3, m4, 0xf0 ; xpos0 xpos1
    paddw                m3, m1
.w8_loop:
    lea                 r5d, [r3+dxq]
    shr                 r3d, 6
    movu                xm0, [tlq+r3*2]
    movu                xm1, [tlq+r3*2+2]
    lea                 r3d, [r5+dxq]
    shr                 r5d, 6
    vinserti128          m0, [tlq+r5*2], 1
    vinserti128          m1, [tlq+r5*2+2], 1
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    psraw                m2, m3, 15
    paddw                m3, m4
    paddw                m0, m1
    vpblendvb            m0, m6, m0, m2
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    sub                  hd, 2
    jz .w8_end
    lea                dstq, [dstq+strideq*2]
    cmp                 r3d, maxbased
    jb .w8_loop
.w8_end_loop:
    mova   [dstq+strideq*0], xm6
    mova   [dstq+strideq*1], xm6
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w8_end_loop
.w8_end:
    RET
.w16_no_intra_edge_filter:
    and            maxbased, 15
    or             maxbased, 16 ; imin(h+15, 31)
    jmp .w16_main
.w16:
    %assign stack_offset org_stack_offset
    ALLOC_STACK         -96, 7
    lea            maxbased, [hq+15]
    test             angled, 0x400
    jnz .w16_no_intra_edge_filter
    call .filter_strength
    test                r5d, r5d
    jz .w16_main
    popcnt              r5d, r5d
    mova                 m0, [tlq-2]            ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    paddw                m1, m0, [tlq+2]        ; 2 3 4 5 6 7 8 9   a b c d e f g h
    cmp                 r5d, 3
    jne .w16_filter_3tap
    vpbroadcastd         m2, [base+pw_3]
    punpcklwd           xm0, xm0
    vpblendd             m0, [tlq-4], 0xfe      ; 0 0 1 2 3 4 5 6   7 8 9 a b c d e
    paddw                m1, [tlq+0]            ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    paddw                m0, m2
    pavgw                m0, [tlq+4]            ; 3 4 5 6 7 8 9 a   b c d e f g h i
    paddw                m0, m1
    psrlw                m0, 2
    movu                 m3, [tlq+32]           ; 2 3 4 5 6 7 8 9   a b c d e f g h
    paddw                m2, [tlq+28]           ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    paddw                m1, m3, [tlq+30]       ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    cmp                  hd, 8
    jl .w16_filter_5tap_h4
    punpckhwd            m3, m3
    je .w16_filter_5tap_h8
    vpblendd             m4, m3, [tlq+36], 0x7f ; 4 5 6 7 8 9 a b   c d e f g h h h
    vpblendd             m3, [tlq+34], 0x7f     ; 3 4 5 6 7 8 9 a   b c d e f g h h
    movzx               r3d, word [tlq+62]
    movzx               r2d, word [tlq+60]
    pavgw                m2, m4
    sub                 r2d, r3d
    paddw                m1, m3
    lea                 r2d, [r2+r3*8+4]
    paddw                m1, m2
    shr                 r2d, 3
    psrlw                m1, 2
    mov            [rsp+66], r3w
    mov            [rsp+64], r2w
    mov                 tlq, rsp
    mov                 r3d, 33
    cmp                  hd, 16
    cmovg          maxbased, r3d
    jmp .w16_filter_end2
.w16_filter_5tap_h8:
    vpblendd            xm4, xm3, [tlq+36], 0x07 ; 4 5 6 7 8 9 9 9
    vpblendd            xm3, [tlq+34], 0x07      ; 3 4 5 6 7 8 9 9
    pavgw               xm2, xm4
    paddw               xm1, xm3
    paddw               xm1, xm2
    psrlw               xm1, 2
    jmp .w16_filter_end2
.w16_filter_5tap_h4:
    pshuflw             xm4, xm3, q3332          ; 4 5 5 5
    pshuflw             xm3, xm3, q3321          ; 3 4 5 5
    pavgw               xm2, xm4
    paddw               xm1, xm3
    paddw               xm1, xm2
    psrlw               xm1, 2
    jmp .w16_filter_end2
.w16_filter_3tap:
    vpbroadcastd         m3, [base+z_filter_k-4+r5*4+12*1]
    vpbroadcastd         m4, [base+z_filter_k-4+r5*4+12*0]
    pmullw               m0, m3, [tlq+0]    ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    movu                 m2, [tlq+32]       ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    pmullw               m1, m4
    pmullw               m3, m2
    paddw                m0, m1
    cmp                  hd, 8
    je .w16_filter_3tap_h8
    jl .w16_filter_3tap_h4
    punpckhwd            m2, m2
    vpblendd             m2, [tlq+34], 0x7f ; 2 3 4 5 6 7 8 9   a b c d e f g g
    jmp .w16_filter_end
.w16_filter_3tap_h4:
    pshuflw             xm2, xm2, q3321     ; 2 3 4 4 _ _ _ _
    jmp .w16_filter_end
.w16_filter_3tap_h8:
    psrldq              xm2, 2
    pshufhw             xm2, xm2, q2210     ; 2 3 4 5 6 7 8 8
.w16_filter_end:
    paddw                m2, [tlq+30]       ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    pmullw               m2, m4
    psrlw                m0, 3
    pxor                 m1, m1
    paddw                m2, m3
    psrlw                m2, 3
    pavgw                m0, m1
    pavgw                m1, m2
.w16_filter_end2:
    mov                 tlq, rsp
    mova           [tlq+ 0], m0
    mova           [tlq+32], m1
.w16_main:
    movd                xm4, dxd
    vpbroadcastw         m6, [tlq+maxbaseq*2]
    shl            maxbased, 6
    vpbroadcastw         m4, xm4
    movd                xm0, maxbased
    mov                 r3d, dxd
    vpbroadcastw         m0, xm0
    paddw                m3, m4, [z_base_inc]
    psubw                m3, m0
.w16_loop:
    lea                 r5d, [r3+dxq]
    shr                 r3d, 6
    movu                 m0, [tlq+r3*2]
    movu                 m1, [tlq+r3*2+2]
    lea                 r3d, [r5+dxq]
    shr                 r5d, 6
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    psraw                m2, m3, 15
    paddw                m3, m4
    paddw                m1, m0
    movu                 m0, [tlq+r5*2]
    vpblendvb            m2, m6, m1, m2
    movu                 m1, [tlq+r5*2+2]
    mova   [dstq+strideq*0], m2
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    psraw                m2, m3, 15
    paddw                m3, m4
    paddw                m0, m1
    vpblendvb            m0, m6, m0, m2
    mova   [dstq+strideq*1], m0
    sub                  hd, 2
    jz .w16_end
    lea                dstq, [dstq+strideq*2]
    cmp                 r3d, maxbased
    jb .w16_loop
.w16_end_loop:
    mova   [dstq+strideq*0], m6
    mova   [dstq+strideq*1], m6
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16_end_loop
.w16_end:
    RET
.w32:
    %assign stack_offset org_stack_offset
    ALLOC_STACK        -160, 8
    lea            maxbased, [hq+31]
    mov                 r3d, 63
    cmp                  hd, 32
    cmova          maxbased, r3d
    test             angled, 0x400
    jnz .w32_main
    vpbroadcastd         m2, [pw_3]
    mova                 m0, [tlq-2]       ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    punpcklwd           xm1, xm0, xm0
    vpblendd             m1, [tlq-4], 0xfe ; 0 0 1 2 3 4 5 6   7 8 9 a b c d e
    paddw                m0, [tlq+0]       ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    paddw                m1, m2
    paddw                m0, [tlq+2]       ; 2 3 4 5 6 7 8 9   a b c d e f g h
    pavgw                m1, [tlq+4]       ; 3 4 5 6 7 8 9 a   b c d e f g h i
    mov                  r3, rsp
    paddw                m0, m1
    lea                 r5d, [maxbaseq-31]
    psrlw                m0, 2
    mova               [r3], m0
.w32_filter_loop:
    mova                 m0, [tlq+30]
    paddw                m1, m2, [tlq+28]
    add                 tlq, 32
    paddw                m0, [tlq+0]
    pavgw                m1, [tlq+4]
    paddw                m0, [tlq+2]
    add                  r3, 32
    paddw                m0, m1
    psrlw                m0, 2
    mova               [r3], m0
    sub                 r5d, 16
    jg .w32_filter_loop
    movu                 m0, [tlq+32]           ; 2 3 4 5 6 7 8 9   a b c d e f g h
    punpckhwd            m1, m0, m0
    paddw                m2, [tlq+28]           ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    paddw                m0, [tlq+30]           ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    jl .w32_filter_h8
    vpblendd             m3, m1, [tlq+36], 0x7f ; 4 5 6 7 8 9 a b   c d e f g h h h
    vpblendd             m1, [tlq+34], 0x7f     ; 3 4 5 6 7 8 9 a   b c d e f g h h
    movzx               r5d, word [tlq+62]
    movzx               r2d, word [tlq+60]
    pavgw                m2, m3
    sub                 r2d, r5d
    paddw                m0, m1
    lea                 r2d, [r2+r5*8+4]
    paddw                m0, m2
    shr                 r2d, 3
    psrlw                m0, 2
    mova            [r3+32], m0
    mov             [r3+66], r5w
    mov             [r3+64], r2w
    mov                 tlq, rsp
    mov                 r3d, 65
    cmp                  hd, 64
    cmove          maxbased, r3d
    jmp .w32_main
.w32_filter_h8:
    vpblendd            xm3, xm1, [tlq+36], 0x07 ; 4 5 6 7 8 9 9 9
    vpblendd            xm1, [tlq+34], 0x07      ; 3 4 5 6 7 8 9 9
    pavgw               xm2, xm3
    paddw               xm0, xm1
    mov                 tlq, rsp
    paddw               xm0, xm2
    psrlw               xm0, 2
    mova            [r3+32], xm0
.w32_main:
    movd                xm4, dxd
    vpbroadcastw         m6, [tlq+maxbaseq*2]
    shl            maxbased, 6
    vpbroadcastw         m4, xm4
    movd                xm0, maxbased
    mov                 r5d, dxd
    vpbroadcastd         m7, [pw_m1024] ; -16 * 64
    vpbroadcastw         m0, xm0
    paddw                m3, m4, [z_base_inc]
    psubw                m3, m0
.w32_loop:
    mov                 r3d, r5d
    shr                 r3d, 6
    movu                 m0, [tlq+r3*2]
    movu                 m1, [tlq+r3*2+2]
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    paddw                m0, m1
    psraw                m1, m3, 15
    vpblendvb            m0, m6, m0, m1
    mova        [dstq+32*0], m0
    movu                 m0, [tlq+r3*2+32]
    movu                 m1, [tlq+r3*2+34]
    add                 r5d, dxd
    psubw                m1, m0
    pmulhrsw             m1, m2
    pcmpgtw              m2, m7, m3
    paddw                m3, m4
    paddw                m0, m1
    vpblendvb            m0, m6, m0, m2
    mova        [dstq+32*1], m0
    dec                  hd
    jz .w32_end
    add                dstq, strideq
    cmp                 r5d, maxbased
    jb .w32_loop
.w32_end_loop:
    mova        [dstq+32*0], m6
    mova        [dstq+32*1], m6
    add                dstq, strideq
    dec                  hd
    jg .w32_end_loop
.w32_end:
    RET
.w64:
    %assign stack_offset org_stack_offset
    ALLOC_STACK        -256, 10
    lea            maxbased, [hq+63]
    test             angled, 0x400
    jnz .w64_main
    vpbroadcastd         m2, [pw_3]
    mova                 m0, [tlq-2]       ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    punpcklwd           xm1, xm0, xm0
    vpblendd             m1, [tlq-4], 0xfe ; 0 0 1 2 3 4 5 6   7 8 9 a b c d e
    paddw                m0, [tlq+0]       ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    paddw                m1, m2
    paddw                m0, [tlq+2]       ; 2 3 4 5 6 7 8 9   a b c d e f g h
    pavgw                m1, [tlq+4]       ; 3 4 5 6 7 8 9 a   b c d e f g h i
    mov                  r3, rsp
    paddw                m0, m1
    lea                 r5d, [hq+32]
    psrlw                m0, 2
    mova               [r3], m0
.w64_filter_loop:
    mova                 m0, [tlq+30]
    paddw                m1, m2, [tlq+28]
    add                 tlq, 32
    paddw                m0, [tlq+0]
    pavgw                m1, [tlq+4]
    paddw                m0, [tlq+2]
    add                  r3, 32
    paddw                m0, m1
    psrlw                m0, 2
    mova               [r3], m0
    sub                 r5d, 16
    jg .w64_filter_loop
    movu                 m0, [tlq+32]           ; 2 3 4 5 6 7 8 9   a b c d e f g h
    punpckhwd            m1, m0, m0
    paddw                m2, [tlq+28]           ; 0 1 2 3 4 5 6 7   8 9 a b c d e f
    paddw                m0, [tlq+30]           ; 1 2 3 4 5 6 7 8   9 a b c d e f g
    vpblendd             m3, m1, [tlq+36], 0x7f ; 4 5 6 7 8 9 a b   c d e f g h h h
    vpblendd             m1, [tlq+34], 0x7f     ; 3 4 5 6 7 8 9 a   b c d e f g h h
    pavgw                m2, m3
    paddw                m0, m1
    paddw                m0, m2
    mov                 tlq, rsp
    psrlw                m0, 2
    mova            [r3+32], m0
.w64_main:
    movd                xm4, dxd
    vpbroadcastw         m6, [tlq+maxbaseq*2]
    shl            maxbased, 6
    vpbroadcastw         m4, xm4
    movd                xm0, maxbased
    mov                 r5d, dxd
    vpbroadcastd         m7, [pw_m1024] ; -16 * 64
    vpbroadcastw         m0, xm0
    paddw                m3, m4, [z_base_inc]
    paddw                m8, m7, m7     ; -32 * 64
    psubw                m3, m0
    paddw                m9, m8, m7     ; -48 * 64
.w64_loop:
    mov                 r3d, r5d
    shr                 r3d, 6
    movu                 m0, [tlq+r3*2]
    movu                 m1, [tlq+r3*2+2]
    pand                 m2, m5, m3
    psllw                m2, 9
    psubw                m1, m0
    pmulhrsw             m1, m2
    paddw                m0, m1
    psraw                m1, m3, 15
    vpblendvb            m0, m6, m0, m1
    mova        [dstq+32*0], m0
    movu                 m0, [tlq+r3*2+32]
    movu                 m1, [tlq+r3*2+34]
    psubw                m1, m0
    pmulhrsw             m1, m2
    paddw                m0, m1
    pcmpgtw              m1, m7, m3
    vpblendvb            m0, m6, m0, m1
    mova        [dstq+32*1], m0
    movu                 m0, [tlq+r3*2+64]
    movu                 m1, [tlq+r3*2+66]
    psubw                m1, m0
    pmulhrsw             m1, m2
    paddw                m0, m1
    pcmpgtw              m1, m8, m3
    vpblendvb            m0, m6, m0, m1
    mova        [dstq+32*2], m0
    movu                 m0, [tlq+r3*2+96]
    movu                 m1, [tlq+r3*2+98]
    add                 r5d, dxd
    psubw                m1, m0
    pmulhrsw             m1, m2
    pcmpgtw              m2, m9, m3
    paddw                m3, m4
    paddw                m0, m1
    vpblendvb            m0, m6, m0, m2
    mova        [dstq+32*3], m0
    dec                  hd
    jz .w64_end
    add                dstq, strideq
    cmp                 r5d, maxbased
    jb .w64_loop
.w64_end_loop:
    mova        [dstq+32*0], m6
    mova        [dstq+32*1], m6
    mova        [dstq+32*2], m6
    mova        [dstq+32*3], m6
    add                dstq, strideq
    dec                  hd
    jg .w64_end_loop
.w64_end:
    RET

%macro FILTER_1BLK 5 ; dst, src, tmp, shuf, bdmax
%ifnum %4
    pshufb             xm%2, xm%4
%else
    pshufb             xm%2, %4
%endif
    vinserti128         m%2, xm%2, 1
    pshufd              m%1, m%2, q0000
    pmaddwd             m%1, m2
    pshufd              m%3, m%2, q1111
    pmaddwd             m%3, m3
    paddd               m%1, m1
    paddd               m%1, m%3
    pshufd              m%3, m%2, q2222
    pmaddwd             m%3, m4
    paddd               m%1, m%3
    pshufd              m%3, m%2, q3333
    pmaddwd             m%3, m5
    paddd               m%1, m%3
    psrad               m%1, 4
    packusdw            m%1, m%1
    pminsw              m%1, m%5
%endmacro

%macro FILTER_2BLK 7 ; dst, src, tmp_dst, tmp_src, tmp, shuf, bdmax
    pshufb              m%2, m%6
    vpermq              m%4, m%2, q3232
    vinserti128         m%2, xm%2, 1
    pshufd              m%1, m%2, q0000
    pshufd              m%3, m%4, q0000
    pmaddwd             m%1, m2
    pmaddwd             m%3, m2
    paddd               m%1, m1
    paddd               m%3, m1
    pshufd              m%5, m%2, q1111
    pmaddwd             m%5, m3
    paddd               m%1, m%5
    pshufd              m%5, m%4, q1111
    pmaddwd             m%5, m3
    paddd               m%3, m%5
    pshufd              m%5, m%2, q2222
    pmaddwd             m%5, m4
    paddd               m%1, m%5
    pshufd              m%5, m%4, q2222
    pmaddwd             m%5, m4
    paddd               m%3, m%5
    pshufd              m%5, m%2, q3333
    pmaddwd             m%5, m5
    paddd               m%1, m%5
    pshufd              m%5, m%4, q3333
    pmaddwd             m%5, m5
    paddd               m%3, m%5
    psrad               m%1, 4
    psrad               m%3, 4
    packusdw            m%1, m%3
    pminsw              m%1, m%7
%endmacro

; The ipred_filter SIMD processes 4x2 blocks in the following order which
; increases parallelism compared to doing things row by row. One redundant
; block is calculated for w8 and w16, two for w32.
;     w4     w8       w16             w32
;     1     1 2     1 2 3 5     1 2 3 5 b c d f
;     2     2 3     2 4 5 7     2 4 5 7 c e f h
;     3     3 4     4 6 7 9     4 6 7 9 e g h j
; ___ 4 ___ 4 5 ___ 6 8 9 a ___ 6 8 9 a g i j k ___
;           5       8           8       i

cglobal ipred_filter_16bpc, 3, 9, 0, dst, stride, tl, w, h, filter
%assign org_stack_offset stack_offset
%define base r6-ipred_filter_16bpc_avx2_table
    lea                  r6, [filter_intra_taps]
    tzcnt                wd, wm
%ifidn filterd, filterm
    movzx           filterd, filterb
%else
    movzx           filterd, byte filterm
%endif
    shl             filterd, 6
    add             filterq, r6
    lea                  r6, [ipred_filter_16bpc_avx2_table]
    vbroadcasti128       m0, [tlq-6]
    movsxd               wq, [r6+wq*4]
    vpbroadcastd         m1, [base+pd_8]
    pmovsxbw             m2, [filterq+16*0]
    pmovsxbw             m3, [filterq+16*1]
    pmovsxbw             m4, [filterq+16*2]
    pmovsxbw             m5, [filterq+16*3]
    add                  wq, r6
    mov                  hd, hm
    jmp                  wq
.w4:
    WIN64_SPILL_XMM      10
    mova                xm8, [base+filter_shuf2]
    vpbroadcastw         m9, r8m ; bitdepth_max
    lea                  r7, [6+hq*2]
    sub                 tlq, r7
    jmp .w4_loop_start
.w4_loop:
    pinsrq              xm0, [tlq+hq*2], 0
    lea                dstq, [dstq+strideq*2]
.w4_loop_start:
    FILTER_1BLK           6, 0, 7, 8, 9
    vextracti128        xm0, m6, 1
    movq   [dstq+strideq*0], xm6
    movq   [dstq+strideq*1], xm0
    sub                  hd, 2
    jg .w4_loop
    RET
ALIGN function_align
.w8:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      16
    vbroadcasti128      m14, [base+filter_shuf3]
    vpbroadcastw        m15, r8m ; bitdepth_max
    FILTER_1BLK          10, 0, 7, [base+filter_shuf2], 15
    vpermq               m6, m10, q1302         ; ____ ____ | ____ 4321
    pslldq               m8, m0, 4
    psrldq               m7, m6, 2
    psrldq               m0, m6, 10
    punpcklwd            m7, m0
    vpblendd             m8, m6, 0x33           ; _0__ 4321 | ____ 4321
    vpblendd             m8, m7, 0x40           ; _056 4321 | ____ 4321
    vpblendd             m8, [tlq-6], 0x30      ; _056 4321 | ____ 4321
    lea                  r7, [16+hq*2]
    sub                 tlq, r7
    jmp .w8_loop_start
.w8_loop:
    vpermq               m8, m9, q1302          ; ____ 4321 | ____ 4321
    vpermq               m6, m9, q2031
    psrldq               m0, m6, 2
    psrldq               m6, 10
    punpcklwd            m6, m0
    vpblendd             m8, m7, 0x80           ; _0__ 4321 | ____ 4321
    vpblendd             m8, m6, 0x40           ; _056 4321 | ____ 4321
    mova                m10, m9
.w8_loop_start:
    vpblendd             m8, [tlq+hq*2], 0x0C   ; _056 4321 | _056 4321
    call .main
    vpblendd            m10, m9, 0xCC
    mova         [dstq+strideq*0], xm10
    vextracti128 [dstq+strideq*1], m10, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    %assign stack_offset stack_offset - stack_size_padded
    ALLOC_STACK          32, 16
    vpbroadcastw        m15, r8m ; bitdepth_max
    sub                  hd, 2
    TAIL_CALL .w16_main, 0
.w16_main:
    mova               xm10, [base+filter_shuf2]
    FILTER_1BLK          13, 0, 6, 10, 15
    vpermq              m12, m13, q3120
    mova               xm14, [base+filter_shuf3]
    vinserti128         m14, [base+filter_shuf1], 1
    vpbroadcastq         m0, [tlq+10]
    vpblendd             m0, [tlq-16], 0x4C     ; ___0 4321 | _056 ____
    psrldq               m6, m12, 8
    vpblendd             m0, m6, 0x03           ; ___0 4321 | _056 4321
    punpcklwd            m6, m12
    vpblendd             m0, m6, 0x80           ; 56_0 4321 | _056 4321
    FILTER_2BLK          12, 0, 6, 7, 8, 14, 15
    vpblendd            m13, m12, 0xCC
    vpermq              m12, m12, q2031         ; 6___ 5___
    psrldq              xm6, xm12, 2
    psrldq              xm8, xm12, 12
    vpblendd            xm6, xm8, 0x01
    pblendw             xm6, [tlq+10], 0xF8     ; 4321 056_
    FILTER_1BLK          11, 6, 8, 10, 15
    vpermq              m11, m11, q3120
    pshufd               m9, m11, q1032
    movu                 m8, [tlq+6]            ; __43 210_ | ____ ____
    pshufd               m8, m8, q3021          ; __0_ 4321 | ____ ____
    pshufhw              m8, m8, q3201          ; ___0 4321 | ____ ____
    vpblendd             m9, m8, 0x70           ; ___0 4321 | ____ 4321
    mova         [dstq+strideq*0], xm13
    vextracti128 [dstq+strideq*1], m13, 1
    lea                  r7, [20+hq*2]
    sub                 tlq, r7
    vpermq               m6, m12, q0123         ; ____ 4321 | ____ 4321
    jmp .w16_loop_start
.w16_loop:
    vpermq              m13, m13, q3322
    vpermq              m11,  m9, q2020
    vpermq               m9,  m9, q1302
    vpermq               m6, m12, q0123
    psrldq               m7, 4
    vpblendd            m13, m10, 0xCC
    vpblendd             m9, m7, 0x40
    mova                 m0, [rsp+8]
    mova         [dstq+strideq*0], xm13
    vextracti128 [dstq+strideq*1], m13, 1
.w16_loop_start:
    mova                m13, m12
    vpblendd             m0, [tlq+hq*2], 0x0C
    psrldq               m7, m12, 8
    punpcklwd            m7, m12
    vpblendd             m0, m6, 0x33           ; ___0 4321 | _056 4321
    vpblendd             m0, m7, 0x80           ; 56_0 4321 | _056 4321
    FILTER_2BLK          10, 0, 6, 7, 8, 14, 15
    vpermq              m12, m10, q2031
    mova            [rsp+8], m0
    psrldq               m8, m11, 8
    psrldq              xm6, xm12, 2
    psrldq              xm7, xm12, 10
    psrldq              xm0, xm13, 2
    punpcklwd            m8, m11
    punpcklwd           xm7, xm6
    vpblendd             m8, m9, 0x73           ; 56_0 4321 | ____ 4321
    vpblendd             m8, m7, 0x04           ; 56_0 4321 | __56 4321
    vpblendd             m8, m0, 0x08           ; 56_0 4321 | _056 4321
    call .main
    vpermq               m8, m11, q3120
    vpblendd             m6, m8, m9, 0xCC
    mova         [dstq+strideq*0+16], xm6
    vextracti128 [dstq+strideq*1+16], m6, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16_loop
    vpermq               m8, m9, q3120
    vextracti128        xm0, m8, 1              ; 4321 ____
    pshufd             xm11, xm11, q1032
    vpblendd            xm0, xm11, 0x02         ; 4321 0___
    psrldq              xm6, xm8, 2
    psrldq              xm7, xm8, 12
    pblendw             xm0, xm6, 0x4           ; 4321 05__
    pblendw             xm0, xm7, 0x2           ; 4321 056_
    FILTER_1BLK           6, 0, 7, [base+filter_shuf2], 15
    vpermq              m12, m13, q1302
    vpblendd            m12, m10, 0xCC
    vpblendd             m9, m6, 0xCC
    mova         [dstq+strideq*0+ 0], xm12
    mova         [dstq+strideq*0+16], xm9
    vextracti128 [dstq+strideq*1+ 0], m12, 1
    vextracti128 [dstq+strideq*1+16], m9, 1
    ret
ALIGN function_align
.w32:
    %assign stack_offset org_stack_offset
    ALLOC_STACK          64, 16
    vpbroadcastw        m15, r8m ; bitdepth_max
    sub                  hd, 2
    lea                  r3, [dstq+32]
    lea                 r5d, [hd*2+20]
    call .w16_main
    mov                dstq, r3
    lea                 tlq, [tlq+r5+32]
    sub                 r5d, 20
    shr                 r5d, 1
    sub                 r5d, 2
    lea                  r4, [dstq+strideq*2-2]
DEFINE_ARGS dst, stride, tl, stride3, left, h
    lea            stride3q, [strideq*3]
    movu                 m8, [tlq-6]                        ; 4321 0___
    mova               xm10, [base+filter_shuf2]
    pinsrw              xm0, xm8, [dstq+strideq*0-2], 2
    pinsrw              xm0, xm0, [dstq+strideq*1-2], 1     ; 4321 056_
    pinsrw              xm9, [leftq+strideq*0], 5
    pinsrw              xm9, [leftq+strideq*1], 4
    FILTER_1BLK          13, 0, 6, 10, 15
    vpermq              m12, m13, q3120
    mova               xm14, [base+filter_shuf3]
    vinserti128         m14, [base+filter_shuf1], 1
    psrldq               m6, m12, 8
    punpcklwd            m7, m6, m12
    vpblendd             m0, m6, 0x03           ; ___0 ____ | _0__ 4321
    vpblendd             m0, m7, 0x80           ; 56_0 ____ | _0__ 4321
    vpblendd             m0, m8, 0x30           ; 56_0 4321 | _0__ 4321
    vpblendd             m0, m9, 0x04           ; 56_0 4321 | _056 4321
    FILTER_2BLK          12, 0, 6, 7, 8, 14, 15
    vpblendd            m13, m12, 0xCC
    pinsrw              xm9, [leftq+strideq*2], 3
    pinsrw              xm9, [leftq+stride3q ], 2
    lea               leftq, [leftq+strideq*4]
    pinsrw              xm9, [leftq+strideq*0], 1
    pinsrw              xm9, [leftq+strideq*1], 0
    movq           [rsp+32], xm9
    mov                 r7d, 1
    pslldq               m8, m9, 4
    vpblendd             m0, m8, 0x0C           ; ___0 ____ | _056 ____
    vpermq              m12, m12, q2031         ; 6___ 5___
    psrldq              xm6, xm12, 2
    psrldq              xm7, xm12, 12
    vpblendd            xm6, xm7, 0x01          ; ____ _56_
    pblendw             xm6, [tlq+10], 0xF8     ; 4321 056_
    FILTER_1BLK          11, 6, 7, 10, 15
    vpermq              m11, m11, q3120
    pshufd               m9, m11, q1032
    vbroadcasti128       m8, [tlq+22]           ; __43 210_ | ____ ____
    pshufd               m8, m8, q3021          ; __0_ 4321 | ____ ____
    pshufhw              m8, m8, q3201          ; ___0 4321 | ____ ____
    vpblendd             m9, m8, 0x70           ; ___0 4321 | ____ 4321
    mova         [dstq+strideq*0], xm13
    vextracti128 [dstq+strideq*1], m13, 1
    vpermq               m6, m12, q0123         ; ____ 4321 | ____ 4321
    jmp .w32_loop_start
.w32_loop_last:
    mova                 m0, [rsp+0]
    jmp .w32_loop
.w32_loop_left:
    mova                 m0, [rsp+0]
    vpblendd             m0, [rsp+32+r7*4-12], 0x0C
    dec                 r7d
    jg .w32_loop
    cmp                  hd, 2
    je .w32_loop
    pinsrw              xm6, [rsp+32], 6
    pinsrw              xm6, [leftq+strideq*2], 5
    pinsrw              xm6, [leftq+stride3q ], 4
    lea               leftq, [leftq+strideq*4]
    pinsrw              xm6, [leftq+strideq*0], 3
    pinsrw              xm6, [leftq+strideq*1], 2
    pinsrw              xm6, [leftq+strideq*2], 1
    pinsrw              xm6, [leftq+stride3q ], 0
    lea               leftq, [leftq+strideq*4]
    movu           [rsp+36], xm6
    pinsrw              xm6, [leftq+strideq*0], 1
    pinsrw              xm6, [leftq+strideq*1], 0
    movd           [rsp+32], xm6
    mov                 r7d, 4
.w32_loop:
    vpermq              m13, m13, q3322
    vpermq              m11,  m9, q2020
    vpermq               m9,  m9, q1302
    vpermq               m6, m12, q0123
    psrldq               m7, 4
    vpblendd            m13, m10, 0xCC
    vpblendd             m9, m7, 0x40           ; ___0 4321 | ____ 4321
    mova         [dstq+strideq*0], xm13
    vextracti128 [dstq+strideq*1], m13, 1
.w32_loop_start:
    mova                m13, m12
    psrldq               m7, m12, 8
    punpcklwd            m7, m12
    vpblendd             m0, m6, 0x33           ; ___0 4321 | _056 4321
    vpblendd             m0, m7, 0x80           ; 56_0 4321 | _056 4321
    FILTER_2BLK          10, 0, 6, 7, 8, 14, 15
    vpermq              m12, m10, q2031
    mova            [rsp+0], m0
    psrldq               m8, m11, 8
    psrldq              xm6, xm12, 2
    psrldq              xm7, xm12, 10
    psrldq              xm0, xm13, 2
    punpcklwd            m8, m11
    punpcklwd           xm7, xm6
    vpblendd             m8, m9, 0x73           ; 56_0 4321 | ____ 4321
    vpblendd             m8, m7, 0x04           ; 56_0 4321 | __56 4321
    vpblendd             m8, m0, 0x08           ; 56_0 4321 | _056 4321
    call .main
    vpermq               m8, m11, q3120
    vpblendd             m6, m8, m9, 0xCC
    mova         [dstq+strideq*0+16], xm6
    vextracti128 [dstq+strideq*1+16], m6, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w32_loop_left
    jz .w32_loop_last
    vpermq               m8, m9, q3120
    vextracti128        xm0, m8, 1              ; 4321 ____
    pshufd             xm11, xm11, q1032
    vpblendd            xm0, xm11, 0x02         ; 4321 0___
    psrldq              xm6, xm8, 2
    psrldq              xm7, xm8, 12
    pblendw             xm0, xm6, 0x4           ; 4321 05__
    pblendw             xm0, xm7, 0x2           ; 4321 056_
    FILTER_1BLK           6, 0, 7, [base+filter_shuf2], 15
    vpermq              m12, m13, q1302
    vpblendd            m12, m10, 0xCC
    vpblendd             m9, m6, 0xCC
    mova         [dstq+strideq*0+ 0], xm12
    mova         [dstq+strideq*0+16], xm9
    vextracti128 [dstq+strideq*1+ 0], m12, 1
    vextracti128 [dstq+strideq*1+16], m9, 1
    RET
.main:
    FILTER_2BLK           9, 8, 6, 7, 0, 14, 15
    ret

%if WIN64
DECLARE_REG_TMP 5
%else
DECLARE_REG_TMP 7
%endif

%macro IPRED_CFL 1 ; ac in, unpacked pixels out
    psignw               m3, m%1, m1
    pabsw               m%1, m%1
    pmulhrsw            m%1, m2
    psignw              m%1, m3
    paddw               m%1, m0
%endmacro

cglobal ipred_cfl_top_16bpc, 3, 7, 8, dst, stride, tl, w, h, ac, alpha
    movifnidn            hd, hm
    add                 tlq, 2
    movd                xm4, wd
    pxor                 m6, m6
    vpbroadcastw         m7, r7m
    pavgw               xm4, xm6
    tzcnt                wd, wd
    movd                xm5, wd
    movu                 m0, [tlq]
    lea                  t0, [ipred_cfl_left_16bpc_avx2_table]
    movsxd               r6, [t0+wq*4]
    add                  r6, t0
    add                  t0, ipred_cfl_splat_16bpc_avx2_table-ipred_cfl_left_16bpc_avx2_table
    movsxd               wq, [t0+wq*4]
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  r6

cglobal ipred_cfl_left_16bpc, 3, 7, 8, dst, stride, tl, w, h, ac, alpha
    mov                  hd, hm ; zero upper half
    sub                 tlq, hq
    movd                xm4, hd
    sub                 tlq, hq
    pxor                 m6, m6
    vpbroadcastw         m7, r7m
    pavgw               xm4, xm6
    tzcnt               r6d, hd
    movd                xm5, r6d
    movu                 m0, [tlq]
    lea                  t0, [ipred_cfl_left_16bpc_avx2_table]
    movsxd               r6, [t0+r6*4]
    add                  r6, t0
    add                  t0, ipred_cfl_splat_16bpc_avx2_table-ipred_cfl_left_16bpc_avx2_table
    tzcnt                wd, wd
    movsxd               wq, [t0+wq*4]
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  r6
.h32:
    paddw                m0, [tlq+32]
.h16:
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
.h8:
    psrldq              xm1, xm0, 8
    paddw               xm0, xm1
.h4:
    punpcklwd           xm0, xm6
    psrlq               xm1, xm0, 32
    paddd               xm0, xm1
    psrldq              xm1, xm0, 8
    paddd               xm0, xm1
    paddd               xm0, xm4
    psrld               xm0, xm5
    vpbroadcastw         m0, xm0
    jmp                  wq

cglobal ipred_cfl_16bpc, 3, 7, 8, dst, stride, tl, w, h, ac, alpha
    movifnidn            hd, hm
    movifnidn            wd, wm
    tzcnt               r6d, hd
    lea                 t0d, [wq+hq]
    movd                xm4, t0d
    tzcnt               t0d, t0d
    movd                xm5, t0d
    lea                  t0, [ipred_cfl_16bpc_avx2_table]
    tzcnt                wd, wd
    movsxd               r6, [t0+r6*4]
    movsxd               wq, [t0+wq*4+4*4]
    psrlw               xm4, 1
    pxor                 m6, m6
    vpbroadcastw         m7, r7m
    add                  r6, t0
    add                  wq, t0
    movifnidn           acq, acmp
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
    punpckhwd           xm1, xm0, xm6
    punpcklwd           xm0, xm6
    paddd               xm0, xm1
    movd                xm1, r6d
    psrld               xm0, 2
    pmulhuw             xm0, xm1
    psrlw               xm0, 1
.w4_end:
    vpbroadcastw         m0, xm0
.s4:
    vpbroadcastw         m1, alpham
    lea                  r6, [strideq*3]
    pabsw                m2, m1
    psllw                m2, 9
.s4_loop:
    mova                 m4, [acq]
    IPRED_CFL             4
    pmaxsw               m4, m6
    pminsw               m4, m7
    vextracti128        xm5, m4, 1
    movq   [dstq+strideq*0], xm4
    movq   [dstq+strideq*2], xm5
    movhps [dstq+strideq*1], xm4
    movhps [dstq+r6       ], xm5
    lea                dstq, [dstq+strideq*4]
    add                 acq, 32
    sub                  hd, 4
    jg .s4_loop
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
    pblendw             xm0, xm6, 0xAA
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
    vpbroadcastw         m0, xm0
.s8:
    vpbroadcastw         m1, alpham
    lea                  r6, [strideq*3]
    pabsw                m2, m1
    psllw                m2, 9
.s8_loop:
    mova                 m4, [acq]
    mova                 m5, [acq+32]
    IPRED_CFL             4
    IPRED_CFL             5
    pmaxsw               m4, m6
    pmaxsw               m5, m6
    pminsw               m4, m7
    pminsw               m5, m7
    mova         [dstq+strideq*0], xm4
    mova         [dstq+strideq*2], xm5
    vextracti128 [dstq+strideq*1], m4, 1
    vextracti128 [dstq+r6       ], m5, 1
    lea                dstq, [dstq+strideq*4]
    add                 acq, 64
    sub                  hd, 4
    jg .s8_loop
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
    punpckhwd           xm1, xm0, xm6
    punpcklwd           xm0, xm6
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
    vpbroadcastw         m1, alpham
    pabsw                m2, m1
    psllw                m2, 9
.s16_loop:
    mova                 m4, [acq]
    mova                 m5, [acq+32]
    IPRED_CFL             4
    IPRED_CFL             5
    pmaxsw               m4, m6
    pmaxsw               m5, m6
    pminsw               m4, m7
    pminsw               m5, m7
    mova   [dstq+strideq*0], m4
    mova   [dstq+strideq*1], m5
    lea                dstq, [dstq+strideq*2]
    add                 acq, 64
    sub                  hd, 2
    jg .s16_loop
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
    punpcklwd           xm1, xm0, xm6
    punpckhwd           xm0, xm6
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
.s32:
    vpbroadcastw         m1, alpham
    pabsw                m2, m1
    psllw                m2, 9
.s32_loop:
    mova                 m4, [acq]
    mova                 m5, [acq+32]
    IPRED_CFL             4
    IPRED_CFL             5
    pmaxsw               m4, m6
    pmaxsw               m5, m6
    pminsw               m4, m7
    pminsw               m5, m7
    mova        [dstq+32*0], m4
    mova        [dstq+32*1], m5
    add                dstq, strideq
    add                 acq, 64
    dec                  hd
    jg .s32_loop
    RET

cglobal ipred_cfl_128_16bpc, 3, 7, 8, dst, stride, tl, w, h, ac, alpha
    mov                 r6d, r7m
    shr                 r6d, 11
    lea                  t0, [ipred_cfl_splat_16bpc_avx2_table]
    tzcnt                wd, wd
    movifnidn            hd, hm
    movsxd               wq, [t0+wq*4]
    vpbroadcastd         m0, [t0-ipred_cfl_splat_16bpc_avx2_table+pw_512+r6*4]
    pxor                 m6, m6
    vpbroadcastw         m7, r7m
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  wq

cglobal ipred_cfl_ac_420_16bpc, 4, 9, 6, ac, ypx, stride, wpad, hpad, w, h, sz, ac_bak
    movifnidn         hpadd, hpadm
    movifnidn            wd, wm
    mov                  hd, hm
    mov                 szd, wd
    mov             ac_bakq, acq
    imul                szd, hd
    shl               hpadd, 2
    sub                  hd, hpadd
    vpbroadcastd         m2, [pw_2]
    pxor                 m4, m4
    cmp                  wd, 8
    jg .w16
    je .w8
DEFINE_ARGS ac, ypx, stride, wpad, hpad, stride3, h, sz, ac_bak
.w4:
    lea            stride3q, [strideq*3]
.w4_loop:
    mova                xm0, [ypxq+strideq*2]
    mova                xm1, [ypxq+stride3q ]
    vinserti128          m0, [ypxq+strideq*0], 1
    vinserti128          m1, [ypxq+strideq*1], 1
    pmaddwd              m0, m2
    pmaddwd              m1, m2
    paddd                m0, m1
    vextracti128        xm1, m0, 1
    paddd                m4, m0
    packssdw            xm1, xm0
    mova              [acq], xm1
    lea                ypxq, [ypxq+strideq*4]
    add                 acq, 16
    sub                  hd, 2
    jg .w4_loop
    test              hpadd, hpadd
    jz .calc_avg
    vpermq               m1, m1, q1111
    pslld               xm0, 2
.w4_hpad_loop:
    mova              [acq], m1
    paddd                m4, m0
    add                 acq, 32
    sub               hpadd, 4
    jg .w4_hpad_loop
    jmp .calc_avg
.w8:
    test              wpadd, wpadd
    jnz .w8_wpad
.w8_loop:
    pmaddwd              m0, m2, [ypxq+strideq*0]
    pmaddwd              m1, m2, [ypxq+strideq*1]
    paddd                m0, m1
    vextracti128        xm1, m0, 1
    paddd                m4, m0
    packssdw            xm1, xm0, xm1
    mova              [acq], xm1
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 16
    dec                  hd
    jg .w8_loop
    jmp .w8_hpad
.w8_wpad:
    pmaddwd             xm0, xm2, [ypxq+strideq*0]
    pmaddwd             xm3, xm2, [ypxq+strideq*1]
    paddd               xm0, xm3
    pshufd              xm3, xm0, q3333
    packssdw            xm1, xm0, xm3
    paddd               xm0, xm3
    paddd               xm4, xm0
    mova              [acq], xm1
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 16
    dec                  hd
    jg .w8_wpad
.w8_hpad:
    test              hpadd, hpadd
    jz .calc_avg
    vinserti128          m1, xm1, 1
    paddd                m0, m0
.w8_hpad_loop:
    paddd                m4, m0
    mova              [acq], m1
    add                 acq, 32
    sub               hpadd, 2
    jg .w8_hpad_loop
    jmp .calc_avg
.w16:
    test              wpadd, wpadd
    jnz .w16_wpad
.w16_loop:
    pmaddwd              m0, m2, [ypxq+strideq*0+ 0]
    pmaddwd              m1, m2, [ypxq+strideq*1+ 0]
    pmaddwd              m3, m2, [ypxq+strideq*0+32]
    pmaddwd              m5, m2, [ypxq+strideq*1+32]
    paddd                m0, m1
    paddd                m3, m5
    packssdw             m1, m0, m3
    paddd                m0, m3
    vpermq               m1, m1, q3120
    paddd                m4, m0
    mova              [acq], m1
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 32
    dec                  hd
    jg .w16_loop
    jmp .w16_hpad
.w16_wpad:
DEFINE_ARGS ac, ypx, stride, wpad, hpad, iptr, h, sz, ac_bak
    lea               iptrq, [ipred_cfl_ac_420_16bpc_avx2_table]
    mov               wpadd, wpadd
    movsxd            wpadq, [iptrq+wpadq*4+4]
    add               iptrq, wpadq
    jmp               iptrq
.w16_wpad_pad3:
    vpbroadcastd         m3, [ypxq+strideq*0+12]
    vpbroadcastd         m5, [ypxq+strideq*1+12]
    vinserti128          m0, m3, [ypxq+strideq*0], 0
    vinserti128          m1, m5, [ypxq+strideq*1], 0
    jmp .w16_wpad_end
.w16_wpad_pad2:
    mova                 m0, [ypxq+strideq*0+ 0]
    mova                 m1, [ypxq+strideq*1+ 0]
    vpbroadcastd         m3, [ypxq+strideq*0+28]
    vpbroadcastd         m5, [ypxq+strideq*1+28]
    jmp .w16_wpad_end
.w16_wpad_pad1:
    mova                 m0, [ypxq+strideq*0+ 0]
    mova                 m1, [ypxq+strideq*1+ 0]
    vpbroadcastd         m3, [ypxq+strideq*0+44]
    vpbroadcastd         m5, [ypxq+strideq*1+44]
    vinserti128          m3, [ypxq+strideq*0+32], 0
    vinserti128          m5, [ypxq+strideq*1+32], 0
.w16_wpad_end:
    pmaddwd              m0, m2
    pmaddwd              m1, m2
    pmaddwd              m3, m2
    pmaddwd              m5, m2
    paddd                m0, m1
    paddd                m3, m5
    packssdw             m1, m0, m3
    paddd                m0, m3
    vpermq               m1, m1, q3120
    paddd                m4, m0
    mova              [acq], m1
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 32
    dec                  hd
    jz .w16_hpad
    jmp               iptrq
.w16_hpad:
    test              hpadd, hpadd
    jz .calc_avg
.w16_hpad_loop:
    mova              [acq], m1
    paddd                m4, m0
    add                 acq, 32
    dec               hpadd
    jg .w16_hpad_loop
.calc_avg:
    vextracti128        xm0, m4, 1
    tzcnt               r1d, szd
    movd                xm3, szd
    paddd               xm0, xm4
    movd                xm2, r1d
    punpckhqdq          xm1, xm0, xm0
    psrld               xm3, 1
    paddd               xm0, xm1
    pshuflw             xm1, xm0, q1032
    paddd               xm0, xm3
    paddd               xm0, xm1
    psrld               xm0, xm2
    vpbroadcastw         m0, xm0
.sub_loop:
    mova                 m1, [ac_bakq]
    psubw                m1, m0
    mova          [ac_bakq], m1
    add             ac_bakq, 32
    sub                 szd, 16
    jg .sub_loop
    RET

cglobal ipred_cfl_ac_422_16bpc, 4, 9, 6, ac, ypx, stride, wpad, hpad, w, h, sz, ac_bak
    movifnidn         hpadd, hpadm
    movifnidn            wd, wm
    mov                  hd, hm
    mov                 szd, wd
    mov             ac_bakq, acq
    imul                szd, hd
    shl               hpadd, 2
    sub                  hd, hpadd
    vpbroadcastd         m2, [pw_4]
    pxor                 m4, m4
    cmp                  wd, 8
    jg .w16
    je .w8
DEFINE_ARGS ac, ypx, stride, wpad, hpad, stride3, h, sz, ac_bak
.w4:
    lea            stride3q, [strideq*3]
.w4_loop:
    mova                xm0, [ypxq+strideq*0]
    mova                xm1, [ypxq+strideq*1]
    vinserti128          m0, [ypxq+strideq*2], 1
    vinserti128          m1, [ypxq+stride3q ], 1
    pmaddwd              m0, m2
    pmaddwd              m1, m2
    paddd                m4, m0
    packssdw             m0, m1
    paddd                m4, m1
    mova              [acq], m0
    lea                ypxq, [ypxq+strideq*4]
    add                 acq, 32
    sub                  hd, 4
    jg .w4_loop
    test              hpadd, hpadd
    jz .calc_avg
    vpermq               m0, m0, q3333
    vextracti128        xm1, m1, 1
    pslld               xm1, 2
.w4_hpad_loop:
    mova              [acq], m0
    paddd                m4, m1
    add                 acq, 32
    sub               hpadd, 4
    jg .w4_hpad_loop
    jmp .calc_avg
.w8:
    test              wpadd, wpadd
    jnz .w8_wpad
.w8_loop:
    pmaddwd              m0, m2, [ypxq+strideq*0]
    pmaddwd              m1, m2, [ypxq+strideq*1]
    paddd                m4, m0
    packssdw             m0, m1
    paddd                m4, m1
    vpermq               m0, m0, q3120
    mova              [acq], m0
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 32
    sub                  hd, 2
    jg .w8_loop
    jmp .w8_hpad
.w8_wpad:
    vpbroadcastd         m0, [ypxq+strideq*0+12]
    vpbroadcastd         m1, [ypxq+strideq*1+12]
    vinserti128          m0, [ypxq+strideq*0+ 0], 0
    vinserti128          m1, [ypxq+strideq*1+ 0], 0
    pmaddwd              m0, m2
    pmaddwd              m1, m2
    paddd                m4, m0
    packssdw             m0, m1
    paddd                m4, m1
    vpermq               m0, m0, q3120
    mova              [acq], m0
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 32
    sub                  hd, 2
    jg .w8_wpad
.w8_hpad:
    test              hpadd, hpadd
    jz .calc_avg
    vpermq               m0, m0, q3232
    paddd                m1, m1
.w8_hpad_loop:
    mova              [acq], m0
    paddd                m4, m1
    add                 acq, 32
    sub               hpadd, 2
    jg .w8_hpad_loop
    jmp .calc_avg
.w16:
    test              wpadd, wpadd
    jnz .w16_wpad
.w16_loop:
    pmaddwd              m3, m2, [ypxq+strideq*0+ 0]
    pmaddwd              m0, m2, [ypxq+strideq*0+32]
    pmaddwd              m1, m2, [ypxq+strideq*1+ 0]
    pmaddwd              m5, m2, [ypxq+strideq*1+32]
    paddd                m4, m3
    packssdw             m3, m0
    paddd                m4, m0
    packssdw             m0, m1, m5
    paddd                m1, m5
    paddd                m4, m1
    vpermq               m3, m3, q3120
    vpermq               m0, m0, q3120
    mova           [acq+ 0], m3
    mova           [acq+32], m0
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 64
    sub                  hd, 2
    jg .w16_loop
    jmp .w16_hpad
.w16_wpad:
DEFINE_ARGS ac, ypx, stride, wpad, hpad, iptr, h, sz, ac_bak
    lea               iptrq, [ipred_cfl_ac_422_16bpc_avx2_table]
    mov               wpadd, wpadd
    movsxd            wpadq, [iptrq+wpadq*4+4]
    add               iptrq, wpadq
    jmp               iptrq
.w16_wpad_pad3:
    vpbroadcastd         m0, [ypxq+strideq*0+12]
    vpbroadcastd         m3, [ypxq+strideq*1+12]
    vinserti128          m5, m0, [ypxq+strideq*0], 0
    vinserti128          m1, m3, [ypxq+strideq*1], 0
    jmp .w16_wpad_end
.w16_wpad_pad2:
    mova                 m5, [ypxq+strideq*0+ 0]
    mova                 m1, [ypxq+strideq*1+ 0]
    vpbroadcastd         m0, [ypxq+strideq*0+28]
    vpbroadcastd         m3, [ypxq+strideq*1+28]
    jmp .w16_wpad_end
.w16_wpad_pad1:
    mova                 m5, [ypxq+strideq*0+ 0]
    mova                 m1, [ypxq+strideq*1+ 0]
    vpbroadcastd         m0, [ypxq+strideq*0+44]
    vpbroadcastd         m3, [ypxq+strideq*1+44]
    vinserti128          m0, [ypxq+strideq*0+32], 0
    vinserti128          m3, [ypxq+strideq*1+32], 0
.w16_wpad_end:
    pmaddwd              m5, m2
    pmaddwd              m1, m2
    pmaddwd              m0, m2
    pmaddwd              m3, m2
    paddd                m4, m5
    packssdw             m5, m0
    paddd                m4, m0
    packssdw             m0, m1, m3
    paddd                m1, m3
    paddd                m4, m1
    vpermq               m5, m5, q3120
    vpermq               m0, m0, q3120
    mova           [acq+ 0], m5
    mova           [acq+32], m0
    lea                ypxq, [ypxq+strideq*2]
    add                 acq, 64
    sub                  hd, 2
    jz .w16_hpad
    jmp               iptrq
.w16_hpad:
    test              hpadd, hpadd
    jz .calc_avg
.w16_hpad_loop:
    mova              [acq], m0
    paddd                m4, m1
    add                 acq, 32
    dec               hpadd
    jg .w16_hpad_loop
.calc_avg:
    vextracti128        xm0, m4, 1
    tzcnt               r1d, szd
    movd                xm2, r1d
    paddd               xm0, xm4
    movd                xm3, szd
    punpckhqdq          xm1, xm0, xm0
    paddd               xm0, xm1
    psrld               xm3, 1
    psrlq               xm1, xm0, 32
    paddd               xm0, xm3
    paddd               xm0, xm1
    psrld               xm0, xm2
    vpbroadcastw         m0, xm0
.sub_loop:
    mova                 m1, [ac_bakq]
    psubw                m1, m0
    mova          [ac_bakq], m1
    add             ac_bakq, 32
    sub                 szd, 16
    jg .sub_loop
    RET

cglobal pal_pred_16bpc, 4, 6, 5, dst, stride, pal, idx, w, h
    vbroadcasti128       m3, [palq]
    lea                  r2, [pal_pred_16bpc_avx2_table]
    tzcnt                wd, wm
    vbroadcasti128       m4, [pal_pred_shuf]
    movifnidn            hd, hm
    movsxd               wq, [r2+wq*4]
    pshufb               m3, m4
    punpckhqdq           m4, m3, m3
    add                  wq, r2
DEFINE_ARGS dst, stride, stride3, idx, w, h
    lea            stride3q, [strideq*3]
    jmp                  wq
.w4:
    mova                xm2, [idxq]
    add                idxq, 16
    pshufb              xm1, xm3, xm2
    pshufb              xm2, xm4, xm2
    punpcklbw           xm0, xm1, xm2
    punpckhbw           xm1, xm2
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*2], xm1
    movhps [dstq+strideq*1], xm0
    movhps [dstq+stride3q ], xm1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w4
    RET
.w8:
    movu                 m2, [idxq] ; only 16-byte alignment
    add                idxq, 32
    pshufb               m1, m3, m2
    pshufb               m2, m4, m2
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova         [dstq+strideq*0], xm0
    mova         [dstq+strideq*1], xm1
    vextracti128 [dstq+strideq*2], m0, 1
    vextracti128 [dstq+stride3q ], m1, 1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w8
    RET
.w16:
    vpermq               m2, [idxq+ 0], q3120
    vpermq               m5, [idxq+32], q3120
    add                idxq, 64
    pshufb               m1, m3, m2
    pshufb               m2, m4, m2
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    pshufb               m1, m3, m5
    pshufb               m2, m4, m5
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova   [dstq+strideq*2], m0
    mova   [dstq+stride3q ], m1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w16
    RET
.w32:
    vpermq               m2, [idxq+ 0], q3120
    vpermq               m5, [idxq+32], q3120
    add                idxq, 64
    pshufb               m1, m3, m2
    pshufb               m2, m4, m2
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova [dstq+strideq*0+ 0], m0
    mova [dstq+strideq*0+32], m1
    pshufb               m1, m3, m5
    pshufb               m2, m4, m5
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova [dstq+strideq*1+ 0], m0
    mova [dstq+strideq*1+32], m1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w32
    RET
.w64:
    vpermq               m2, [idxq+ 0], q3120
    vpermq               m5, [idxq+32], q3120
    add                idxq, 64
    pshufb               m1, m3, m2
    pshufb               m2, m4, m2
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova          [dstq+ 0], m0
    mova          [dstq+32], m1
    pshufb               m1, m3, m5
    pshufb               m2, m4, m5
    punpcklbw            m0, m1, m2
    punpckhbw            m1, m2
    mova          [dstq+64], m0
    mova          [dstq+96], m1
    add                 dstq, strideq
    dec                   hd
    jg .w64
    RET

%endif
