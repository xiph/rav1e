; Copyright © 2018, VideoLAN and dav1d authors
; Copyright © 2018, Two Orioles, LLC
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
        db %1-128, 127-%1
        %rotate 1
    %endrep
%endmacro

; sm_weights[], but modified to precalculate x and 256-x with offsets to
; enable efficient use of pmaddubsw (which requires signed values)
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

ipred_v_shuf: db  0,  1,  0,  1,  4,  5,  4,  5,  8,  9,  8,  9, 12, 13, 12, 13
              db  2,  3,  2,  3,  6,  7,  6,  7, 10, 11, 10, 11, 14, 15, 14, 15
ipred_h_shuf: db  7,  7,  7,  7,  3,  3,  3,  3,  5,  5,  5,  5,  1,  1,  1,  1
              db  6,  6,  6,  6,  2,  2,  2,  2,  4,  4,  4,  4,  0,  0,  0,  0

pb_1:   times 4 db 1
pb_128: times 4 db 128
pw_128: times 2 dw 128
pw_255: times 2 dw 255

pb_127_m127: times 2 db 127, -127

%macro JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - 2*4)
    %xdefine %%base mangle(private_prefix %+ _%1_%2)
    %%table:
    %rep %0 - 2
        dd %%base %+ .%3 - (%%table - 2*4)
        %rotate 1
    %endrep
%endmacro

%define ipred_dc_splat_avx2_table (ipred_dc_avx2_table + 10*4)
%define ipred_cfl_splat_avx2_table (ipred_cfl_avx2_table + 8*4)

JMP_TABLE ipred_smooth,   avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_v, avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_smooth_h, avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_paeth,    avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_dc,       avx2, h4, h8, h16, h32, h64, w4, w8, w16, w32, w64, \
                                s4-10*4, s8-10*4, s16-10*4, s32-10*4, s64-10*4
JMP_TABLE ipred_dc_left,  avx2, h4, h8, h16, h32, h64
JMP_TABLE ipred_h,        avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_cfl,      avx2, h4, h8, h16, h32, w4, w8, w16, w32, \
                                s4-8*4, s8-8*4, s16-8*4, s32-8*4
JMP_TABLE ipred_cfl_left, avx2, h4, h8, h16, h32

SECTION .text

INIT_YMM avx2
cglobal ipred_dc_top, 3, 7, 6, dst, stride, tl, w, h
    lea                  r5, [ipred_dc_left_avx2_table]
    tzcnt                wd, wm
    inc                 tlq
    movu                 m0, [tlq]
    movifnidn            hd, hm
    mov                 r6d, 0x8000
    shrx                r6d, r6d, wd
    movd                xm3, r6d
    movsxd               r6, [r5+wq*4]
    pcmpeqd              m2, m2
    pmaddubsw            m0, m2
    add                  r6, r5
    add                  r5, ipred_dc_splat_avx2_table-ipred_dc_left_avx2_table
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6

cglobal ipred_dc_left, 3, 7, 6, dst, stride, tl, w, h, stride3
    mov                  hd, hm ; zero upper half
    tzcnt               r6d, hd
    sub                 tlq, hq
    tzcnt                wd, wm
    movu                 m0, [tlq]
    mov                 r5d, 0x8000
    shrx                r5d, r5d, r6d
    movd                xm3, r5d
    lea                  r5, [ipred_dc_left_avx2_table]
    movsxd               r6, [r5+r6*4]
    pcmpeqd              m2, m2
    pmaddubsw            m0, m2
    add                  r6, r5
    add                  r5, ipred_dc_splat_avx2_table-ipred_dc_left_avx2_table
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    jmp                  r6
.h64:
    movu                 m1, [tlq+32] ; unaligned when jumping here from dc_top
    pmaddubsw            m1, m2
    paddw                m0, m1
.h32:
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
.h16:
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
.h8:
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
.h4:
    pmaddwd             xm0, xm2
    pmulhrsw            xm0, xm3
    lea            stride3q, [strideq*3]
    vpbroadcastb         m0, xm0
    mova                 m1, m0
    jmp                  wq

cglobal ipred_dc, 3, 7, 6, dst, stride, tl, w, h, stride3
    movifnidn            hd, hm
    movifnidn            wd, wm
    tzcnt               r6d, hd
    lea                 r5d, [wq+hq]
    movd                xm4, r5d
    tzcnt               r5d, r5d
    movd                xm5, r5d
    lea                  r5, [ipred_dc_avx2_table]
    tzcnt                wd, wd
    movsxd               r6, [r5+r6*4]
    movsxd               wq, [r5+wq*4+5*4]
    pcmpeqd              m3, m3
    psrlw               xm4, 1
    add                  r6, r5
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  r6
.h4:
    movd                xm0, [tlq-4]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w4:
    movd                xm1, [tlq+1]
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    cmp                  hd, 4
    jg .w4_mul
    psrlw               xm0, 3
    jmp .w4_end
.w4_mul:
    punpckhqdq          xm1, xm0, xm0
    lea                 r2d, [hq*2]
    mov                 r6d, 0x55563334
    paddw               xm0, xm1
    shrx                r6d, r6d, r2d
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    movd                xm1, r6d
    psrlw               xm0, 2
    pmulhuw             xm0, xm1
.w4_end:
    vpbroadcastb        xm0, xm0
.s4:
    movd   [dstq+strideq*0], xm0
    movd   [dstq+strideq*1], xm0
    movd   [dstq+strideq*2], xm0
    movd   [dstq+stride3q ], xm0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s4
    RET
ALIGN function_align
.h8:
    movq                xm0, [tlq-8]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w8:
    movq                xm1, [tlq+1]
    vextracti128        xm2, m0, 1
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm2
    punpckhqdq          xm2, xm0, xm0
    paddw               xm0, xm2
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 8
    je .w8_end
    mov                 r6d, 0x5556
    mov                 r2d, 0x3334
    cmp                  hd, 32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
.w8_end:
    vpbroadcastb        xm0, xm0
.s8:
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm0
    movq   [dstq+strideq*2], xm0
    movq   [dstq+stride3q ], xm0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s8
    RET
ALIGN function_align
.h16:
    mova                xm0, [tlq-16]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w16:
    movu                xm1, [tlq+1]
    vextracti128        xm2, m0, 1
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm2
    paddw               xm0, xm1
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 16
    je .w16_end
    mov                 r6d, 0x5556
    mov                 r2d, 0x3334
    test                 hb, 8|32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
.w16_end:
    vpbroadcastb        xm0, xm0
.s16:
    mova   [dstq+strideq*0], xm0
    mova   [dstq+strideq*1], xm0
    mova   [dstq+strideq*2], xm0
    mova   [dstq+stride3q ], xm0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s16
    RET
ALIGN function_align
.h32:
    mova                 m0, [tlq-32]
    pmaddubsw            m0, m3
    jmp                  wq
.w32:
    movu                 m1, [tlq+1]
    pmaddubsw            m1, m3
    paddw                m0, m1
    vextracti128        xm1, m0, 1
    psubw               xm0, xm4
    paddw               xm0, xm1
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 32
    je .w32_end
    lea                 r2d, [hq*2]
    mov                 r6d, 0x33345556
    shrx                r6d, r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
.w32_end:
    vpbroadcastb         m0, xm0
.s32:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m0
    mova   [dstq+strideq*2], m0
    mova   [dstq+stride3q ], m0
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .s32
    RET
ALIGN function_align
.h64:
    mova                 m0, [tlq-64]
    mova                 m1, [tlq-32]
    pmaddubsw            m0, m3
    pmaddubsw            m1, m3
    paddw                m0, m1
    jmp                  wq
.w64:
    movu                 m1, [tlq+ 1]
    movu                 m2, [tlq+33]
    pmaddubsw            m1, m3
    pmaddubsw            m2, m3
    paddw                m0, m1
    paddw                m0, m2
    vextracti128        xm1, m0, 1
    psubw               xm0, xm4
    paddw               xm0, xm1
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 64
    je .w64_end
    mov                 r6d, 0x33345556
    shrx                r6d, r6d, hd
    movd                xm1, r6d
    pmulhuw             xm0, xm1
.w64_end:
    vpbroadcastb         m0, xm0
    mova                 m1, m0
.s64:
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
    jg .s64
    RET

cglobal ipred_dc_128, 2, 7, 6, dst, stride, tl, w, h, stride3
    lea                  r5, [ipred_dc_splat_avx2_table]
    tzcnt                wd, wm
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    vpbroadcastd         m0, [r5-ipred_dc_splat_avx2_table+pb_128]
    mova                 m1, m0
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq

cglobal ipred_v, 3, 7, 6, dst, stride, tl, w, h, stride3
    lea                  r5, [ipred_dc_splat_avx2_table]
    tzcnt                wd, wm
    movu                 m0, [tlq+ 1]
    movu                 m1, [tlq+33]
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq

%macro IPRED_H 2 ; w, store_type
    vpbroadcastb         m0, [tlq-1]
    vpbroadcastb         m1, [tlq-2]
    vpbroadcastb         m2, [tlq-3]
    sub                 tlq, 4
    vpbroadcastb         m3, [tlq+0]
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

INIT_XMM avx2
cglobal ipred_h, 3, 6, 4, dst, stride, tl, w, h, stride3
    lea                  r5, [ipred_h_avx2_table]
    tzcnt                wd, wm
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    add                  wq, r5
    lea            stride3q, [strideq*3]
    jmp                  wq
.w4:
    IPRED_H               4, d
.w8:
    IPRED_H               8, q
.w16:
    IPRED_H              16, a
INIT_YMM avx2
.w32:
    IPRED_H              32, a
.w64:
    vpbroadcastb         m0, [tlq-1]
    vpbroadcastb         m1, [tlq-2]
    vpbroadcastb         m2, [tlq-3]
    sub                 tlq, 4
    vpbroadcastb         m3, [tlq+0]
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
    jg .w64
    RET

%macro PAETH 2 ; top, ldiff
    pavgb                m1, m%1, m3 ; Calculating tldiff normally requires
    pxor                 m0, m%1, m3 ; 10-bit intermediates, but we can do it
    pand                 m0, m4      ; in 8-bit with some tricks which avoids
    psubusb              m2, m5, m1  ; having to unpack everything to 16-bit.
    psubb                m1, m0
    psubusb              m1, m5
    por                  m1, m2
    paddusb              m1, m1
    por                  m1, m0      ; min(tldiff, 255)
    psubusb              m2, m5, m3
    psubusb              m0, m3, m5
    por                  m2, m0      ; tdiff
    pminub               m2, m%2
    pcmpeqb              m0, m%2, m2 ; ldiff <= tdiff
    vpblendvb            m0, m%1, m3, m0
    pminub               m1, m2
    pcmpeqb              m1, m2      ; ldiff <= tldiff || tdiff <= tldiff
    vpblendvb            m0, m5, m0, m1
%endmacro

cglobal ipred_paeth, 3, 6, 9, dst, stride, tl, w, h
%define base r5-ipred_paeth_avx2_table
    lea                  r5, [ipred_paeth_avx2_table]
    tzcnt                wd, wm
    vpbroadcastb         m5, [tlq]   ; topleft
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    vpbroadcastd         m4, [base+pb_1]
    add                  wq, r5
    jmp                  wq
.w4:
    vpbroadcastd         m6, [tlq+1] ; top
    mova                 m8, [base+ipred_h_shuf]
    lea                  r3, [strideq*3]
    psubusb              m7, m5, m6
    psubusb              m0, m6, m5
    por                  m7, m0      ; ldiff
.w4_loop:
    sub                 tlq, 8
    vpbroadcastq         m3, [tlq]
    pshufb               m3, m8      ; left
    PAETH                 6, 7
    vextracti128        xm1, m0, 1
    movd   [dstq+strideq*0], xm0
    movd   [dstq+strideq*1], xm1
    pextrd [dstq+strideq*2], xm0, 2
    pextrd [dstq+r3       ], xm1, 2
    cmp                  hd, 4
    je .ret
    lea                dstq, [dstq+strideq*4]
    pextrd [dstq+strideq*0], xm0, 1
    pextrd [dstq+strideq*1], xm1, 1
    pextrd [dstq+strideq*2], xm0, 3
    pextrd [dstq+r3       ], xm1, 3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 8
    jg .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    vpbroadcastq         m6, [tlq+1]
    mova                 m8, [base+ipred_h_shuf]
    lea                  r3, [strideq*3]
    psubusb              m7, m5, m6
    psubusb              m0, m6, m5
    por                  m7, m0
.w8_loop:
    sub                 tlq, 4
    vpbroadcastd         m3, [tlq]
    pshufb               m3, m8
    PAETH                 6, 7
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    vbroadcasti128       m6, [tlq+1]
    mova                xm8, xm4 ; lower half = 1, upper half = 0
    psubusb              m7, m5, m6
    psubusb              m0, m6, m5
    por                  m7, m0
.w16_loop:
    sub                 tlq, 2
    vpbroadcastd         m3, [tlq]
    pshufb               m3, m8
    PAETH                 6, 7
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16_loop
    RET
ALIGN function_align
.w32:
    movu                 m6, [tlq+1]
    psubusb              m7, m5, m6
    psubusb              m0, m6, m5
    por                  m7, m0
.w32_loop:
    dec                 tlq
    vpbroadcastb         m3, [tlq]
    PAETH                 6, 7
    mova             [dstq], m0
    add                dstq, strideq
    dec                  hd
    jg .w32_loop
    RET
ALIGN function_align
.w64:
    movu                 m6, [tlq+ 1]
    movu                 m7, [tlq+33]
%if WIN64
    movaps              r4m, xmm9
%endif
    psubusb              m8, m5, m6
    psubusb              m0, m6, m5
    psubusb              m9, m5, m7
    psubusb              m1, m7, m5
    por                  m8, m0
    por                  m9, m1
.w64_loop:
    dec                 tlq
    vpbroadcastb         m3, [tlq]
    PAETH                 6, 8
    mova        [dstq+32*0], m0
    PAETH                 7, 9
    mova        [dstq+32*1], m0
    add                dstq, strideq
    dec                  hd
    jg .w64_loop
%if WIN64
    movaps             xmm9, r4m
%endif
    RET

%macro SMOOTH 6 ; src[1-2], mul[1-2], add[1-2]
    ; w * a         = (w - 128) * a + 128 * a
    ; (256 - w) * b = (127 - w) * b + 129 * b
    pmaddubsw            m0, m%3, m%1
    pmaddubsw            m1, m%4, m%2
    paddw                m0, m%5
    paddw                m1, m%6
    psrlw                m0, 8
    psrlw                m1, 8
    packuswb             m0, m1
%endmacro

cglobal ipred_smooth_v, 3, 7, 0, dst, stride, tl, w, h, weights
%define base r6-ipred_smooth_v_avx2_table
    lea                  r6, [ipred_smooth_v_avx2_table]
    tzcnt                wd, wm
    mov                  hd, hm
    movsxd               wq, [r6+wq*4]
    vpbroadcastd         m0, [base+pb_127_m127]
    vpbroadcastd         m1, [base+pw_128]
    lea            weightsq, [base+smooth_weights+hq*4]
    neg                  hq
    vpbroadcastb         m5, [tlq+hq] ; bottom
    add                  wq, r6
    jmp                  wq
.w4:
    vpbroadcastd         m2, [tlq+1]
    punpcklbw            m2, m5 ; top, bottom
    mova                 m5, [base+ipred_v_shuf]
    lea                  r3, [strideq*3]
    punpckldq            m4, m5, m5
    punpckhdq            m5, m5
    pmaddubsw            m3, m2, m0
    paddw                m1, m2 ;   1 * top + 256 * bottom + 128, overflow is ok
    paddw                m3, m1 ; 128 * top + 129 * bottom + 128
.w4_loop:
    vbroadcasti128       m1, [weightsq+hq*2]
    pshufb               m0, m1, m4
    pshufb               m1, m5
    SMOOTH                0, 1, 2, 2, 3, 3
    vextracti128        xm1, m0, 1
    movd   [dstq+strideq*0], xm0
    movd   [dstq+strideq*1], xm1
    pextrd [dstq+strideq*2], xm0, 1
    pextrd [dstq+r3       ], xm1, 1
    cmp                  hd, -4
    je .ret
    lea                dstq, [dstq+strideq*4]
    pextrd [dstq+strideq*0], xm0, 2
    pextrd [dstq+strideq*1], xm1, 2
    pextrd [dstq+strideq*2], xm0, 3
    pextrd [dstq+r3       ], xm1, 3
    lea                dstq, [dstq+strideq*4]
    add                  hq, 8
    jl .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    vpbroadcastq         m2, [tlq+1]
    punpcklbw            m2, m5
    mova                 m5, [base+ipred_v_shuf]
    lea                  r3, [strideq*3]
    pshufd               m4, m5, q0000
    pshufd               m5, m5, q1111
    pmaddubsw            m3, m2, m0
    paddw                m1, m2
    paddw                m3, m1
.w8_loop:
    vpbroadcastq         m1, [weightsq+hq*2]
    pshufb               m0, m1, m4
    pshufb               m1, m5
    SMOOTH                0, 1, 2, 2, 3, 3
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea                dstq, [dstq+strideq*4]
    add                  hq, 4
    jl .w8_loop
    RET
ALIGN function_align
.w16:
    WIN64_SPILL_XMM       7
    vbroadcasti128       m3, [tlq+1]
    mova                 m6, [base+ipred_v_shuf]
    punpcklbw            m2, m3, m5
    punpckhbw            m3, m5
    pmaddubsw            m4, m2, m0
    pmaddubsw            m5, m3, m0
    paddw                m0, m1, m2
    paddw                m1, m3
    paddw                m4, m0
    paddw                m5, m1
.w16_loop:
    vpbroadcastd         m1, [weightsq+hq*2]
    pshufb               m1, m6
    SMOOTH                1, 1, 2, 3, 4, 5
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea                dstq, [dstq+strideq*2]
    add                  hq, 2
    jl .w16_loop
    RET
ALIGN function_align
.w32:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM       6
    movu                 m3, [tlq+1]
    punpcklbw            m2, m3, m5
    punpckhbw            m3, m5
    pmaddubsw            m4, m2, m0
    pmaddubsw            m5, m3, m0
    paddw                m0, m1, m2
    paddw                m1, m3
    paddw                m4, m0
    paddw                m5, m1
.w32_loop:
    vpbroadcastw         m1, [weightsq+hq*2]
    SMOOTH                1, 1, 2, 3, 4, 5
    mova             [dstq], m0
    add                dstq, strideq
    inc                  hq
    jl .w32_loop
    RET
ALIGN function_align
.w64:
    WIN64_SPILL_XMM      11
    movu                 m4, [tlq+ 1]
    movu                 m8, [tlq+33]
    punpcklbw            m3, m4, m5
    punpckhbw            m4, m5
    punpcklbw            m7, m8, m5
    punpckhbw            m8, m5
    pmaddubsw            m5, m3, m0
    pmaddubsw            m6, m4, m0
    pmaddubsw            m9, m7, m0
    pmaddubsw           m10, m8, m0
    paddw                m2, m1, m3
    paddw                m5, m2
    paddw                m2, m1, m4
    paddw                m6, m2
    paddw                m0, m1, m7
    paddw                m9, m0
    paddw                m1, m8
    paddw               m10, m1
.w64_loop:
    vpbroadcastw         m2, [weightsq+hq*2]
    SMOOTH                2, 2, 3, 4, 5, 6
    mova        [dstq+32*0], m0
    SMOOTH                2, 2, 7, 8, 9, 10
    mova        [dstq+32*1], m0
    add                dstq, strideq
    inc                  hq
    jl .w64_loop
    RET

%macro SETUP_STACK_FRAME 3 ; stack_size, regs_used, xmm_regs_used
    %assign stack_offset 0
    %assign stack_size_padded 0
    %assign regs_used %2
    %xdefine rstk rsp
    SETUP_STACK_POINTER %1
    %if regs_used != %2 && WIN64
        PUSH r%2
    %endif
    ALLOC_STACK %1, %3
%endmacro

cglobal ipred_smooth_h, 3, 7, 0, dst, stride, tl, w, h
%define base r6-ipred_smooth_h_avx2_table
    lea                  r6, [ipred_smooth_h_avx2_table]
    mov                  wd, wm
    vpbroadcastb         m3, [tlq+wq] ; right
    tzcnt                wd, wd
    mov                  hd, hm
    movsxd               wq, [r6+wq*4]
    vpbroadcastd         m4, [base+pb_127_m127]
    vpbroadcastd         m5, [base+pw_128]
    add                  wq, r6
    jmp                  wq
.w4:
    WIN64_SPILL_XMM       8
    vpbroadcastq         m6, [base+smooth_weights+4*2]
    mova                 m7, [base+ipred_h_shuf]
    sub                 tlq, 8
    sub                 tlq, hq
    lea                  r3, [strideq*3]
.w4_loop:
    vpbroadcastq         m2, [tlq+hq]
    pshufb               m2, m7
    punpcklbw            m1, m2, m3 ; left, right
    punpckhbw            m2, m3
    pmaddubsw            m0, m1, m4 ; 127 * left - 127 * right
    paddw                m0, m1     ; 128 * left + 129 * right
    pmaddubsw            m1, m6
    paddw                m1, m5
    paddw                m0, m1
    pmaddubsw            m1, m2, m4
    paddw                m1, m2
    pmaddubsw            m2, m6
    paddw                m2, m5
    paddw                m1, m2
    psrlw                m0, 8
    psrlw                m1, 8
    packuswb             m0, m1
    vextracti128        xm1, m0, 1
    movd   [dstq+strideq*0], xm0
    movd   [dstq+strideq*1], xm1
    pextrd [dstq+strideq*2], xm0, 2
    pextrd [dstq+r3       ], xm1, 2
    cmp                  hd, 4
    je .ret
    lea                dstq, [dstq+strideq*4]
    pextrd [dstq+strideq*0], xm0, 1
    pextrd [dstq+strideq*1], xm1, 1
    pextrd [dstq+strideq*2], xm0, 3
    pextrd [dstq+r3       ], xm1, 3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 8
    jg .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM       8
    vbroadcasti128       m6, [base+smooth_weights+8*2]
    mova                 m7, [base+ipred_h_shuf]
    sub                 tlq, 4
    lea                  r3, [strideq*3]
    sub                 tlq, hq
.w8_loop:
    vpbroadcastd         m2, [tlq+hq]
    pshufb               m2, m7
    punpcklbw            m1, m2, m3
    punpckhbw            m2, m3
    pmaddubsw            m0, m1, m4
    paddw                m0, m1
    pmaddubsw            m1, m6
    paddw                m1, m5
    paddw                m0, m1
    pmaddubsw            m1, m2, m4
    paddw                m1, m2
    pmaddubsw            m2, m6
    paddw                m2, m5
    paddw                m1, m2
    psrlw                m0, 8
    psrlw                m1, 8
    packuswb             m0, m1
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    SETUP_STACK_FRAME  32*4, 7, 8
    lea                  r3, [rsp+64*2-4]
    call .prep ; only worthwhile for for w16 and above
    sub                 tlq, 2
    vpbroadcastd        xm6, [base+pb_1]
    mova                xm7, [base+ipred_v_shuf+16]
    vinserti128          m7, [base+ipred_v_shuf+ 0], 1
    vbroadcasti128       m4, [base+smooth_weights+16*2]
    vbroadcasti128       m5, [base+smooth_weights+16*3]
.w16_loop:
    vpbroadcastd         m1, [tlq+hq]
    vpbroadcastd         m2, [r3+hq*2]
    pshufb               m1, m6
    punpcklbw            m1, m3
    pshufb               m2, m7
    SMOOTH                4, 5, 1, 1, 2, 2
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16_loop
    RET
ALIGN function_align
.w32:
    SETUP_STACK_FRAME  32*4, 7, 6
    lea                  r3, [rsp+64*2-2]
    call .prep
    dec                 tlq
    mova                xm4, [base+smooth_weights+16*4]
    vinserti128          m4, [base+smooth_weights+16*6], 1
    mova                xm5, [base+smooth_weights+16*5]
    vinserti128          m5, [base+smooth_weights+16*7], 1
.w32_loop:
    vpbroadcastb         m1, [tlq+hq]
    punpcklbw            m1, m3
    vpbroadcastw         m2, [r3+hq*2]
    SMOOTH                4, 5, 1, 1, 2, 2
    mova             [dstq], m0
    add                dstq, strideq
    dec                  hd
    jg .w32_loop
    RET
ALIGN function_align
.w64:
    SETUP_STACK_FRAME  32*4, 7, 9
    lea                  r3, [rsp+64*2-2]
    call .prep
    add                  r6, smooth_weights+16*15-ipred_smooth_h_avx2_table
    dec                 tlq
    mova                xm5, [r6-16*7]
    vinserti128          m5, [r6-16*5], 1
    mova                xm6, [r6-16*6]
    vinserti128          m6, [r6-16*4], 1
    mova                xm7, [r6-16*3]
    vinserti128          m7, [r6-16*1], 1
    mova                xm8, [r6-16*2]
    vinserti128          m8, [r6-16*0], 1
.w64_loop:
    vpbroadcastb         m2, [tlq+hq]
    punpcklbw            m2, m3
    vpbroadcastw         m4, [r3+hq*2]
    SMOOTH                5, 6, 2, 2, 4, 4
    mova        [dstq+32*0], m0
    SMOOTH                7, 8, 2, 2, 4, 4
    mova        [dstq+32*1], m0
    add                dstq, strideq
    dec                  hd
    jg .w64_loop
    RET
ALIGN function_align
.prep:
    vpermq               m2, [tlq-32*1], q3120
    punpckhbw            m1, m2, m3
    punpcklbw            m2, m3
    pmaddubsw            m0, m1, m4 ; 127 * left - 127 * right
    paddw                m1, m5     ;   1 * left + 256 * right + 128
    paddw                m0, m1     ; 128 * left + 129 * right + 128
    pmaddubsw            m1, m2, m4
    paddw                m2, m5
    paddw                m1, m2
    vpermq               m2, [tlq-32*2], q3120
    mova [rsp+gprsize+32*3], m0
    mova [rsp+gprsize+32*2], m1
    punpckhbw            m1, m2, m3
    punpcklbw            m2, m3
    pmaddubsw            m0, m1, m4
    paddw                m1, m5
    paddw                m0, m1
    pmaddubsw            m1, m2, m4
    paddw                m2, m5
    paddw                m1, m2
    mova [rsp+gprsize+32*1], m0
    mova [rsp+gprsize+32*0], m1
    sub                  r3, hq
    sub                 tlq, hq
    sub                  r3, hq
    ret

%macro SMOOTH_2D_END 6 ; src[1-2], mul[1-2], add[1-2]
    pmaddubsw            m0, m%3, m%1
    pmaddubsw            m1, m%4, m%2
%ifnum %5
    paddw                m0, m%5
%else
    paddw                m0, %5
%endif
%ifnum %6
    paddw                m1, m%6
%else
    paddw                m1, %6
%endif
    pavgw                m0, m2
    pavgw                m1, m3
    psrlw                m0, 8
    psrlw                m1, 8
    packuswb             m0, m1
%endmacro

cglobal ipred_smooth, 3, 7, 0, dst, stride, tl, w, h, v_weights
%define base r6-ipred_smooth_avx2_table
    lea                  r6, [ipred_smooth_avx2_table]
    mov                  wd, wm
    vpbroadcastb         m4, [tlq+wq] ; right
    tzcnt                wd, wd
    mov                  hd, hm
    mov                  r5, tlq
    sub                  r5, hq
    movsxd               wq, [r6+wq*4]
    vpbroadcastd         m5, [base+pb_127_m127]
    vpbroadcastb         m0, [r5] ; bottom
    vpbroadcastd         m3, [base+pw_255]
    add                  wq, r6
    lea          v_weightsq, [base+smooth_weights+hq*2]
    jmp                  wq
.w4:
    WIN64_SPILL_XMM      12
    mova                m10, [base+ipred_h_shuf]
    vpbroadcastq        m11, [base+smooth_weights+4*2]
    mova                 m7, [base+ipred_v_shuf]
    vpbroadcastd         m8, [tlq+1]
    sub                 tlq, 8
    lea                  r3, [strideq*3]
    sub                 tlq, hq
    punpcklbw            m8, m0 ; top, bottom
    pshufd               m6, m7, q2200
    pshufd               m7, m7, q3311
    pmaddubsw            m9, m8, m5
    paddw                m3, m8 ;   1 * top + 255 * bottom + 255
    paddw                m9, m3 ; 128 * top + 129 * bottom + 255
.w4_loop:
    vpbroadcastq         m1, [tlq+hq]
    pshufb               m1, m10
    punpcklbw            m0, m1, m4 ; left, right
    punpckhbw            m1, m4
    pmaddubsw            m2, m0, m5 ; 127 * left - 127 * right
    pmaddubsw            m3, m1, m5
    paddw                m2, m0     ; 128 * left + 129 * right
    paddw                m3, m1
    pmaddubsw            m0, m11
    pmaddubsw            m1, m11
    paddw                m2, m0
    paddw                m3, m1
    vbroadcasti128       m1, [v_weightsq]
    add          v_weightsq, 16
    pshufb               m0, m1, m6
    pshufb               m1, m7
    SMOOTH_2D_END         0, 1, 8, 8, 9, 9
    vextracti128        xm1, m0, 1
    movd   [dstq+strideq*0], xm0
    movd   [dstq+strideq*1], xm1
    pextrd [dstq+strideq*2], xm0, 2
    pextrd [dstq+r3       ], xm1, 2
    cmp                  hd, 4
    je .ret
    lea                dstq, [dstq+strideq*4]
    pextrd [dstq+strideq*0], xm0, 1
    pextrd [dstq+strideq*1], xm1, 1
    pextrd [dstq+strideq*2], xm0, 3
    pextrd [dstq+r3       ], xm1, 3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 8
    jg .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      12
    mova                m10, [base+ipred_h_shuf]
    vbroadcasti128      m11, [base+smooth_weights+8*2]
    mova                 m7, [base+ipred_v_shuf]
    vpbroadcastq         m8, [tlq+1]
    sub                 tlq, 4
    lea                  r3, [strideq*3]
    sub                 tlq, hq
    punpcklbw            m8, m0
    pshufd               m6, m7, q0000
    pshufd               m7, m7, q1111
    pmaddubsw            m9, m8, m5
    paddw                m3, m8
    paddw                m9, m3
.w8_loop:
    vpbroadcastd         m1, [tlq+hq]
    pshufb               m1, m10
    punpcklbw            m0, m1, m4
    punpckhbw            m1, m4
    pmaddubsw            m2, m0, m5
    pmaddubsw            m3, m1, m5
    paddw                m2, m0
    paddw                m3, m1
    pmaddubsw            m0, m11
    pmaddubsw            m1, m11
    paddw                m2, m0
    paddw                m3, m1
    vpbroadcastq         m1, [v_weightsq]
    add          v_weightsq, 8
    pshufb               m0, m1, m6
    pshufb               m1, m7
    SMOOTH_2D_END         0, 1, 8, 8, 9, 9
    vextracti128        xm1, m0, 1
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xm1
    movhps [dstq+strideq*2], xm0
    movhps [dstq+r3       ], xm1
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 4
    jg .w8_loop
    RET
ALIGN function_align
.w16:
    SETUP_STACK_FRAME  32*4, 7, 14
    vbroadcasti128      m11, [tlq+1]
    lea                  r3, [rsp+64*2-4]
    punpcklbw           m10, m11, m0 ; top, bottom
    punpckhbw           m11, m0
    call .prep_v
    sub                 tlq, 2
    pmaddubsw           m12, m10, m5
    pmaddubsw           m13, m11, m5
    vpbroadcastd        xm5, [base+pb_1]
    mova                 m9, [base+ipred_v_shuf]
    vbroadcasti128       m6, [base+smooth_weights+16*2]
    vbroadcasti128       m7, [base+smooth_weights+16*3]
    vpermq               m8, m9, q1032
    paddw                m0, m10, m3
    paddw                m3, m11
    paddw               m12, m0
    paddw               m13, m3
.w16_loop:
    vpbroadcastd         m3, [tlq+hq]
    vpbroadcastd         m0, [r3+hq*2]
    vpbroadcastd         m1, [v_weightsq]
    add          v_weightsq, 4
    pshufb               m3, m5
    punpcklbw            m3, m4 ; left, right
    pmaddubsw            m2, m3, m6
    pmaddubsw            m3, m7
    pshufb               m0, m8
    pshufb               m1, m9
    paddw                m2, m0
    paddw                m3, m0
    SMOOTH_2D_END         1, 1, 10, 11, 12, 13
    mova         [dstq+strideq*0], xm0
    vextracti128 [dstq+strideq*1], m0, 1
    lea                dstq, [dstq+strideq*2]
    sub                  hd, 2
    jg .w16_loop
    RET
ALIGN function_align
.w32:
    SETUP_STACK_FRAME  32*4, 7, 11
    movu                 m8, [tlq+1]
    lea                  r3, [rsp+64*2-2]
    punpcklbw            m7, m8, m0
    punpckhbw            m8, m0
    call .prep_v
    dec                 tlq
    pmaddubsw            m9, m7, m5
    pmaddubsw           m10, m8, m5
    mova                xm5, [base+smooth_weights+16*4]
    vinserti128          m5, [base+smooth_weights+16*6], 1
    mova                xm6, [base+smooth_weights+16*5]
    vinserti128          m6, [base+smooth_weights+16*7], 1
    paddw                m0, m7, m3
    paddw                m3, m8
    paddw                m9, m0
    paddw               m10, m3
.w32_loop:
    vpbroadcastb         m3, [tlq+hq]
    punpcklbw            m3, m4
    vpbroadcastw         m0, [r3+hq*2]
    vpbroadcastw         m1, [v_weightsq]
    add          v_weightsq, 2
    pmaddubsw            m2, m3, m5
    pmaddubsw            m3, m6
    paddw                m2, m0
    paddw                m3, m0
    SMOOTH_2D_END         1, 1, 7, 8, 9, 10
    mova             [dstq], m0
    add                dstq, strideq
    dec                  hd
    jg .w32_loop
    RET
ALIGN function_align
.w64:
    SETUP_STACK_FRAME  32*8, 7, 16
    movu                m13, [tlq+1 ]
    movu                m15, [tlq+33]
    add                  r6, smooth_weights+16*15-ipred_smooth_avx2_table
    lea                  r3, [rsp+64*2-2]
    punpcklbw           m12, m13, m0
    punpckhbw           m13, m0
    punpcklbw           m14, m15, m0
    punpckhbw           m15, m0
    call .prep_v
    dec                 tlq
    pmaddubsw            m0, m12, m5
    pmaddubsw            m1, m13, m5
    pmaddubsw            m2, m14, m5
    pmaddubsw            m5, m15, m5
    mova                xm8, [r6-16*7]
    vinserti128          m8, [r6-16*5], 1
    mova                xm9, [r6-16*6]
    vinserti128          m9, [r6-16*4], 1
    mova               xm10, [r6-16*3]
    vinserti128         m10, [r6-16*1], 1
    mova               xm11, [r6-16*2]
    vinserti128         m11, [r6-16*0], 1
    lea                  r6, [rsp+32*4]
    paddw                m0, m3
    paddw                m1, m3
    paddw                m2, m3
    paddw                m3, m5
    paddw                m0, m12
    paddw                m1, m13
    paddw                m2, m14
    paddw                m3, m15
    mova          [r6+32*0], m0
    mova          [r6+32*1], m1
    mova          [r6+32*2], m2
    mova          [r6+32*3], m3
.w64_loop:
    vpbroadcastb         m5, [tlq+hq]
    punpcklbw            m5, m4
    vpbroadcastw         m6, [r3+hq*2]
    vpbroadcastw         m7, [v_weightsq]
    add          v_weightsq, 2
    pmaddubsw            m2, m5, m8
    pmaddubsw            m3, m5, m9
    paddw                m2, m6
    paddw                m3, m6
    SMOOTH_2D_END         7, 7, 12, 13, [r6+32*0], [r6+32*1]
    mova        [dstq+32*0], m0
    pmaddubsw            m2, m5, m10
    pmaddubsw            m3, m5, m11
    paddw                m2, m6
    paddw                m3, m6
    SMOOTH_2D_END         7, 7, 14, 15, [r6+32*2], [r6+32*3]
    mova        [dstq+32*1], m0
    add                dstq, strideq
    dec                  hd
    jg .w64_loop
    RET
ALIGN function_align
.prep_v:
    vpermq               m2, [tlq-32*1], q3120
    punpckhbw            m1, m2, m4
    punpcklbw            m2, m4
    pmaddubsw            m0, m1, m5 ; 127 * left - 127 * right
    paddw                m0, m1     ; 128 * left + 129 * right
    pmaddubsw            m1, m2, m5
    paddw                m1, m2
    vpermq               m2, [tlq-32*2], q3120
    mova [rsp+gprsize+32*3], m0
    mova [rsp+gprsize+32*2], m1
    punpckhbw            m1, m2, m4
    punpcklbw            m2, m4
    pmaddubsw            m0, m1, m5
    paddw                m0, m1
    pmaddubsw            m1, m2, m5
    paddw                m1, m2
    mova [rsp+gprsize+32*1], m0
    mova [rsp+gprsize+32*0], m1
    sub                  r3, hq
    sub                 tlq, hq
    sub                  r3, hq
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

cglobal ipred_cfl_top, 3, 7, 6, dst, stride, tl, w, h, ac, alpha
    lea                  t0, [ipred_cfl_left_avx2_table]
    tzcnt                wd, wm
    inc                 tlq
    movu                 m0, [tlq]
    movifnidn            hd, hm
    mov                 r6d, 0x8000
    shrx                r6d, r6d, wd
    movd                xm3, r6d
    movsxd               r6, [t0+wq*4]
    pcmpeqd              m2, m2
    pmaddubsw            m0, m2
    add                  r6, t0
    add                  t0, ipred_cfl_splat_avx2_table-ipred_cfl_left_avx2_table
    movsxd               wq, [t0+wq*4]
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  r6

cglobal ipred_cfl_left, 3, 7, 6, dst, stride, tl, w, h, ac, alpha
    mov                  hd, hm ; zero upper half
    tzcnt               r6d, hd
    sub                 tlq, hq
    tzcnt                wd, wm
    movu                 m0, [tlq]
    mov                 t0d, 0x8000
    shrx                t0d, t0d, r6d
    movd                xm3, t0d
    lea                  t0, [ipred_cfl_left_avx2_table]
    movsxd               r6, [t0+r6*4]
    pcmpeqd              m2, m2
    pmaddubsw            m0, m2
    add                  r6, t0
    add                  t0, ipred_cfl_splat_avx2_table-ipred_cfl_left_avx2_table
    movsxd               wq, [t0+wq*4]
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  r6
.h32:
    vextracti128        xm1, m0, 1
    paddw               xm0, xm1
.h16:
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
.h8:
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
.h4:
    pmaddwd             xm0, xm2
    pmulhrsw            xm0, xm3
    vpbroadcastw         m0, xm0
    jmp                  wq

cglobal ipred_cfl, 3, 7, 6, dst, stride, tl, w, h, ac, alpha
    movifnidn            hd, hm
    movifnidn            wd, wm
    tzcnt               r6d, hd
    lea                 t0d, [wq+hq]
    movd                xm4, t0d
    tzcnt               t0d, t0d
    movd                xm5, t0d
    lea                  t0, [ipred_cfl_avx2_table]
    tzcnt                wd, wd
    movsxd               r6, [t0+r6*4]
    movsxd               wq, [t0+wq*4+4*4]
    pcmpeqd              m3, m3
    psrlw               xm4, 1
    add                  r6, t0
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  r6
.h4:
    movd                xm0, [tlq-4]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w4:
    movd                xm1, [tlq+1]
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    cmp                  hd, 4
    jg .w4_mul
    psrlw               xm0, 3
    jmp .w4_end
.w4_mul:
    punpckhqdq          xm1, xm0, xm0
    lea                 r2d, [hq*2]
    mov                 r6d, 0x55563334
    paddw               xm0, xm1
    shrx                r6d, r6d, r2d
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    movd                xm1, r6d
    psrlw               xm0, 2
    pmulhuw             xm0, xm1
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
    packuswb             m4, m4
    vextracti128        xm5, m4, 1
    movd   [dstq+strideq*0], xm4
    pextrd [dstq+strideq*1], xm4, 1
    movd   [dstq+strideq*2], xm5
    pextrd [dstq+r6       ], xm5, 1
    lea                dstq, [dstq+strideq*4]
    add                 acq, 32
    sub                  hd, 4
    jg .s4_loop
    RET
ALIGN function_align
.h8:
    movq                xm0, [tlq-8]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w8:
    movq                xm1, [tlq+1]
    vextracti128        xm2, m0, 1
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm2
    punpckhqdq          xm2, xm0, xm0
    paddw               xm0, xm2
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 8
    je .w8_end
    mov                 r6d, 0x5556
    mov                 r2d, 0x3334
    cmp                  hd, 32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
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
    packuswb             m4, m5
    vextracti128        xm5, m4, 1
    movq   [dstq+strideq*0], xm4
    movq   [dstq+strideq*1], xm5
    movhps [dstq+strideq*2], xm4
    movhps [dstq+r6       ], xm5
    lea                dstq, [dstq+strideq*4]
    add                 acq, 64
    sub                  hd, 4
    jg .s8_loop
    RET
ALIGN function_align
.h16:
    mova                xm0, [tlq-16]
    pmaddubsw           xm0, xm3
    jmp                  wq
.w16:
    movu                xm1, [tlq+1]
    vextracti128        xm2, m0, 1
    pmaddubsw           xm1, xm3
    psubw               xm0, xm4
    paddw               xm0, xm2
    paddw               xm0, xm1
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 16
    je .w16_end
    mov                 r6d, 0x5556
    mov                 r2d, 0x3334
    test                 hb, 8|32
    cmovz               r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
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
    packuswb             m4, m5
    vpermq               m4, m4, q3120
    mova         [dstq+strideq*0], xm4
    vextracti128 [dstq+strideq*1], m4, 1
    lea                dstq, [dstq+strideq*2]
    add                 acq, 64
    sub                  hd, 2
    jg .s16_loop
    RET
ALIGN function_align
.h32:
    mova                 m0, [tlq-32]
    pmaddubsw            m0, m3
    jmp                  wq
.w32:
    movu                 m1, [tlq+1]
    pmaddubsw            m1, m3
    paddw                m0, m1
    vextracti128        xm1, m0, 1
    psubw               xm0, xm4
    paddw               xm0, xm1
    punpckhqdq          xm1, xm0, xm0
    paddw               xm0, xm1
    psrlq               xm1, xm0, 32
    paddw               xm0, xm1
    pmaddwd             xm0, xm3
    psrlw               xm0, xm5
    cmp                  hd, 32
    je .w32_end
    lea                 r2d, [hq*2]
    mov                 r6d, 0x33345556
    shrx                r6d, r6d, r2d
    movd                xm1, r6d
    pmulhuw             xm0, xm1
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
    packuswb             m4, m5
    vpermq               m4, m4, q3120
    mova             [dstq], m4
    add                dstq, strideq
    add                 acq, 64
    dec                  hd
    jg .s32_loop
    RET

cglobal ipred_cfl_128, 3, 7, 6, dst, stride, tl, w, h, ac, alpha
    lea                  t0, [ipred_cfl_splat_avx2_table]
    tzcnt                wd, wm
    movifnidn            hd, hm
    movsxd               wq, [t0+wq*4]
    vpbroadcastd         m0, [t0-ipred_cfl_splat_avx2_table+pw_128]
    add                  wq, t0
    movifnidn           acq, acmp
    jmp                  wq

%endif
