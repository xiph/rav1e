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

wiener_shufA:  db  2,  3,  4,  5,  4,  5,  6,  7,  6,  7,  8,  9,  8,  9, 10, 11
wiener_shufB:  db  6,  7,  4,  5,  8,  9,  6,  7, 10, 11,  8,  9, 12, 13, 10, 11
wiener_shufC:  db  6,  7,  8,  9,  8,  9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15
wiener_shufD:  db  2,  3, -1, -1,  4,  5, -1, -1,  6,  7, -1, -1,  8,  9, -1, -1
wiener_shufE:  db  0,  1,  8,  9,  2,  3, 10, 11,  4,  5, 12, 13,  6,  7, 14, 15
wiener_lshuf5: db  4,  5,  4,  5,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
wiener_lshuf7: db  8,  9,  8,  9,  8,  9,  8,  9,  8,  9, 10, 11, 12, 13, 14, 15
pb_0to31:      db  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
               db 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31

wiener_hshift: dw 4, 4, 1, 1
wiener_vshift: dw 1024, 1024, 4096, 4096
wiener_round:  dd 1049600, 1048832

pb_m10_m9:     times 2 db -10, -9
pb_m6_m5:      times 2 db  -6, -5
pb_m2_m1:      times 2 db  -2, -1
pb_2_3:        times 2 db   2,  3
pb_6_7:        times 2 db   6,  7
pd_m262128     dd -262128

SECTION .text

%macro REPX 2-*
    %xdefine %%f(x) %1
%rep %0 - 1
    %rotate 1
    %%f(%1)
%endrep
%endmacro

DECLARE_REG_TMP 4, 9, 7, 11, 12, 13, 14 ; wiener ring buffer pointers

INIT_YMM avx2
cglobal wiener_filter7_16bpc, 5, 15, 16, -384*12-16, dst, dst_stride, left, lpf, \
                                                     lpf_stride, w, edge, flt, h
%define base t4-wiener_hshift
    mov           fltq, fltmp
    mov          edged, r8m
    movifnidn       wd, wm
    mov             hd, r6m
    mov            t3d, r9m ; pixel_max
    vbroadcasti128  m6, [wiener_shufA]
    vpbroadcastd   m12, [fltq+ 0] ; x0 x1
    lea             t4, [wiener_hshift]
    vbroadcasti128  m7, [wiener_shufB]
    add             wd, wd
    vpbroadcastd   m13, [fltq+ 4] ; x2 x3
    shr            t3d, 11
    vbroadcasti128  m8, [wiener_shufC]
    add           lpfq, wq
    vbroadcasti128  m9, [wiener_shufD]
    lea             t1, [rsp+wq+16]
    vpbroadcastd   m14, [fltq+16] ; y0 y1
    add           dstq, wq
    vpbroadcastd   m15, [fltq+20] ; y2 y3
    neg             wq
    vpbroadcastd    m0, [base+wiener_hshift+t3*4]
    vpbroadcastd   m10, [base+wiener_round+t3*4]
    vpbroadcastd   m11, [base+wiener_vshift+t3*4]
    pmullw         m12, m0 ; upshift filter coefs to make the
    pmullw         m13, m0 ; horizontal downshift constant
    test         edgeb, 4 ; LR_HAVE_TOP
    jz .no_top
    call .h_top
    add           lpfq, lpf_strideq
    mov             t6, t1
    mov             t5, t1
    add             t1, 384*2
    call .h_top
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov             t4, t1
    add             t1, 384*2
    mov      [rsp+8*1], lpf_strideq
    add             r7, lpf_strideq
    mov      [rsp+8*0], r7 ; below
    call .h
    mov             t3, t1
    mov             t2, t1
    dec             hd
    jz .v1
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    mov             t2, t1
    dec             hd
    jz .v2
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    dec             hd
    jz .v3
.main:
    lea             t0, [t1+384*2]
.main_loop:
    call .hv
    dec             hd
    jnz .main_loop
    test         edgeb, 8 ; LR_HAVE_BOTTOM
    jz .v3
    mov           lpfq, [rsp+8*0]
    call .hv_bottom
    add           lpfq, [rsp+8*1]
    call .hv_bottom
.v1:
    call .v
    RET
.no_top:
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov      [rsp+8*1], lpf_strideq
    lea             r7, [r7+lpf_strideq*2]
    mov      [rsp+8*0], r7
    call .h
    mov             t6, t1
    mov             t5, t1
    mov             t4, t1
    mov             t3, t1
    mov             t2, t1
    dec             hd
    jz .v1
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    mov             t2, t1
    dec             hd
    jz .v2
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    dec             hd
    jz .v3
    lea             t0, [t1+384*2]
    call .hv
    dec             hd
    jz .v3
    add             t0, 384*8
    call .hv
    dec             hd
    jnz .main
.v3:
    call .v
.v2:
    call .v
    jmp .v1
.extend_right:
    movd           xm1, r10d
    vpbroadcastd    m0, [pb_6_7]
    movu            m2, [pb_0to31]
    vpbroadcastb    m1, xm1
    psubb           m0, m1
    pminub          m0, m2
    pshufb          m3, m0
    vpbroadcastd    m0, [pb_m2_m1]
    psubb           m0, m1
    pminub          m0, m2
    pshufb          m4, m0
    vpbroadcastd    m0, [pb_m10_m9]
    psubb           m0, m1
    pminub          m0, m2
    pshufb          m5, m0
    ret
.h:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
    movq           xm3, [leftq]
    vpblendd        m3, [lpfq+r10-8], 0xfc
    add          leftq, 8
    jmp .h_main
.h_extend_left:
    vbroadcasti128  m3, [lpfq+r10] ; avoid accessing memory located
    mova            m4, [lpfq+r10] ; before the start of the buffer
    shufpd          m3, m4, 0x05
    pshufb          m3, [wiener_lshuf7]
    jmp .h_main2
.h_top:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
.h_loop:
    movu            m3, [lpfq+r10-8]
.h_main:
    mova            m4, [lpfq+r10+0]
.h_main2:
    movu            m5, [lpfq+r10+8]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .h_have_right
    cmp           r10d, -36
    jl .h_have_right
    call .extend_right
.h_have_right:
    pshufb          m0, m3, m6
    pshufb          m1, m4, m7
    paddw           m0, m1
    pshufb          m3, m8
    pmaddwd         m0, m12
    pshufb          m1, m4, m9
    paddw           m3, m1
    pshufb          m1, m4, m6
    pmaddwd         m3, m13
    pshufb          m2, m5, m7
    paddw           m1, m2
    vpbroadcastd    m2, [pd_m262128] ; (1 << 4) - (1 << 18)
    pshufb          m4, m8
    pmaddwd         m1, m12
    pshufb          m5, m9
    paddw           m4, m5
    pmaddwd         m4, m13
    paddd           m0, m2
    paddd           m1, m2
    paddd           m0, m3
    paddd           m1, m4
    psrad           m0, 4
    psrad           m1, 4
    packssdw        m0, m1
    psraw           m0, 1
    mova      [t1+r10], m0
    add            r10, 32
    jl .h_loop
    ret
ALIGN function_align
.hv:
    add           lpfq, dst_strideq
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
    movq           xm3, [leftq]
    vpblendd        m3, [lpfq+r10-8], 0xfc
    add          leftq, 8
    jmp .hv_main
.hv_extend_left:
    movu            m3, [lpfq+r10-8]
    pshufb          m3, [wiener_lshuf7]
    jmp .hv_main
.hv_bottom:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
.hv_loop:
    movu            m3, [lpfq+r10-8]
.hv_main:
    mova            m4, [lpfq+r10+0]
    movu            m5, [lpfq+r10+8]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .hv_have_right
    cmp           r10d, -36
    jl .hv_have_right
    call .extend_right
.hv_have_right:
    pshufb          m0, m3, m6
    pshufb          m1, m4, m7
    paddw           m0, m1
    pshufb          m3, m8
    pmaddwd         m0, m12
    pshufb          m1, m4, m9
    paddw           m3, m1
    pshufb          m1, m4, m6
    pmaddwd         m3, m13
    pshufb          m2, m5, m7
    paddw           m1, m2
    vpbroadcastd    m2, [pd_m262128]
    pshufb          m4, m8
    pmaddwd         m1, m12
    pshufb          m5, m9
    paddw           m4, m5
    pmaddwd         m4, m13
    paddd           m0, m2
    paddd           m1, m2
    mova            m2, [t4+r10]
    paddw           m2, [t2+r10]
    mova            m5, [t3+r10]
    paddd           m0, m3
    paddd           m1, m4
    psrad           m0, 4
    psrad           m1, 4
    packssdw        m0, m1
    mova            m4, [t5+r10]
    paddw           m4, [t1+r10]
    psraw           m0, 1
    paddw           m3, m0, [t6+r10]
    mova      [t0+r10], m0
    punpcklwd       m0, m2, m5
    pmaddwd         m0, m15
    punpckhwd       m2, m5
    pmaddwd         m2, m15
    punpcklwd       m1, m3, m4
    pmaddwd         m1, m14
    punpckhwd       m3, m4
    pmaddwd         m3, m14
    paddd           m0, m10
    paddd           m2, m10
    paddd           m0, m1
    paddd           m2, m3
    psrad           m0, 5
    psrad           m2, 5
    packusdw        m0, m2
    pmulhuw         m0, m11
    mova    [dstq+r10], m0
    add            r10, 32
    jl .hv_loop
    mov             t6, t5
    mov             t5, t4
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    mov             t1, t0
    mov             t0, t6
    add           dstq, dst_strideq
    ret
.v:
    mov            r10, wq
.v_loop:
    mova            m1, [t4+r10]
    paddw           m1, [t2+r10]
    mova            m2, [t3+r10]
    mova            m4, [t1+r10]
    paddw           m3, m4, [t6+r10]
    paddw           m4, [t5+r10]
    punpcklwd       m0, m1, m2
    pmaddwd         m0, m15
    punpckhwd       m1, m2
    pmaddwd         m1, m15
    punpcklwd       m2, m3, m4
    pmaddwd         m2, m14
    punpckhwd       m3, m4
    pmaddwd         m3, m14
    paddd           m0, m10
    paddd           m1, m10
    paddd           m0, m2
    paddd           m1, m3
    psrad           m0, 5
    psrad           m1, 5
    packusdw        m0, m1
    pmulhuw         m0, m11
    mova    [dstq+r10], m0
    add            r10, 32
    jl .v_loop
    mov             t6, t5
    mov             t5, t4
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    add           dstq, dst_strideq
    ret
cglobal wiener_filter5_16bpc, 5, 13, 16, 384*8+16, dst, dst_stride, left, lpf, \
                                                   lpf_stride, w, edge, flt, h
%define base t4-wiener_hshift
    mov           fltq, fltmp
    mov          edged, r8m
    movifnidn       wd, wm
    mov             hd, r6m
    mov            t3d, r9m ; pixel_max
    vbroadcasti128  m5, [wiener_shufE]
    vpbroadcastw   m11, [fltq+ 2] ; x1
    vbroadcasti128  m6, [wiener_shufB]
    lea             t4, [wiener_hshift]
    vbroadcasti128  m7, [wiener_shufD]
    add             wd, wd
    vpbroadcastd   m12, [fltq+ 4] ; x2 x3
    shr            t3d, 11
    vpbroadcastd    m8, [pd_m262128] ; (1 << 4) - (1 << 18)
    add           lpfq, wq
    lea             t1, [rsp+wq+16]
    vpbroadcastw   m13, [fltq+18] ; y1
    add           dstq, wq
    vpbroadcastd   m14, [fltq+20] ; y2 y3
    neg             wq
    vpbroadcastd    m0, [base+wiener_hshift+t3*4]
    vpbroadcastd    m9, [base+wiener_round+t3*4]
    vpbroadcastd   m10, [base+wiener_vshift+t3*4]
    movu          xm15, [wiener_lshuf5]
    pmullw         m11, m0
    vinserti128    m15, [pb_0to31], 1
    pmullw         m12, m0
    test         edgeb, 4 ; LR_HAVE_TOP
    jz .no_top
    call .h_top
    add           lpfq, lpf_strideq
    mov             t4, t1
    add             t1, 384*2
    call .h_top
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov             t3, t1
    add             t1, 384*2
    mov      [rsp+8*1], lpf_strideq
    add             r7, lpf_strideq
    mov      [rsp+8*0], r7 ; below
    call .h
    mov             t2, t1
    dec             hd
    jz .v1
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    dec             hd
    jz .v2
.main:
    mov             t0, t4
.main_loop:
    call .hv
    dec             hd
    jnz .main_loop
    test         edgeb, 8 ; LR_HAVE_BOTTOM
    jz .v2
    mov           lpfq, [rsp+8*0]
    call .hv_bottom
    add           lpfq, [rsp+8*1]
    call .hv_bottom
.end:
    RET
.no_top:
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov      [rsp+8*1], lpf_strideq
    lea             r7, [r7+lpf_strideq*2]
    mov      [rsp+8*0], r7
    call .h
    mov             t4, t1
    mov             t3, t1
    mov             t2, t1
    dec             hd
    jz .v1
    add           lpfq, dst_strideq
    add             t1, 384*2
    call .h
    dec             hd
    jz .v2
    lea             t0, [t1+384*2]
    call .hv
    dec             hd
    jz .v2
    add             t0, 384*6
    call .hv
    dec             hd
    jnz .main
.v2:
    call .v
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    add           dstq, dst_strideq
.v1:
    call .v
    jmp .end
.extend_right:
    movd           xm2, r10d
    vpbroadcastd    m0, [pb_2_3]
    vpbroadcastd    m1, [pb_m6_m5]
    vpbroadcastb    m2, xm2
    psubb           m0, m2
    psubb           m1, m2
    movu            m2, [pb_0to31]
    pminub          m0, m2
    pminub          m1, m2
    pshufb          m3, m0
    pshufb          m4, m1
    ret
.h:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
    movd           xm3, [leftq+4]
    vpblendd        m3, [lpfq+r10-4], 0xfe
    add          leftq, 8
    jmp .h_main
.h_extend_left:
    vbroadcasti128  m4, [lpfq+r10] ; avoid accessing memory located
    mova            m3, [lpfq+r10] ; before the start of the buffer
    palignr         m3, m4, 12
    pshufb          m3, m15
    jmp .h_main
.h_top:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
.h_loop:
    movu            m3, [lpfq+r10-4]
.h_main:
    movu            m4, [lpfq+r10+4]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .h_have_right
    cmp           r10d, -34
    jl .h_have_right
    call .extend_right
.h_have_right:
    pshufb          m0, m3, m5
    pmaddwd         m0, m11
    pshufb          m1, m4, m5
    pmaddwd         m1, m11
    pshufb          m2, m3, m6
    pshufb          m3, m7
    paddw           m2, m3
    pshufb          m3, m4, m6
    pmaddwd         m2, m12
    pshufb          m4, m7
    paddw           m3, m4
    pmaddwd         m3, m12
    paddd           m0, m8
    paddd           m1, m8
    paddd           m0, m2
    paddd           m1, m3
    psrad           m0, 4
    psrad           m1, 4
    packssdw        m0, m1
    psraw           m0, 1
    mova      [t1+r10], m0
    add            r10, 32
    jl .h_loop
    ret
ALIGN function_align
.hv:
    add           lpfq, dst_strideq
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
    movd           xm3, [leftq+4]
    vpblendd        m3, [lpfq+r10-4], 0xfe
    add          leftq, 8
    jmp .hv_main
.hv_extend_left:
    movu            m3, [lpfq+r10-4]
    pshufb          m3, m15
    jmp .hv_main
.hv_bottom:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
.hv_loop:
    movu            m3, [lpfq+r10-4]
.hv_main:
    movu            m4, [lpfq+r10+4]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .hv_have_right
    cmp           r10d, -34
    jl .hv_have_right
    call .extend_right
.hv_have_right:
    pshufb          m0, m3, m5
    pmaddwd         m0, m11
    pshufb          m1, m4, m5
    pmaddwd         m1, m11
    pshufb          m2, m3, m6
    pshufb          m3, m7
    paddw           m2, m3
    pshufb          m3, m4, m6
    pmaddwd         m2, m12
    pshufb          m4, m7
    paddw           m3, m4
    pmaddwd         m3, m12
    paddd           m0, m8
    paddd           m1, m8
    paddd           m0, m2
    mova            m2, [t3+r10]
    paddw           m2, [t1+r10]
    paddd           m1, m3
    mova            m4, [t2+r10]
    punpckhwd       m3, m2, m4
    pmaddwd         m3, m14
    punpcklwd       m2, m4
    mova            m4, [t4+r10]
    psrad           m0, 4
    psrad           m1, 4
    packssdw        m0, m1
    pmaddwd         m2, m14
    psraw           m0, 1
    mova      [t0+r10], m0
    punpckhwd       m1, m0, m4
    pmaddwd         m1, m13
    punpcklwd       m0, m4
    pmaddwd         m0, m13
    paddd           m3, m9
    paddd           m2, m9
    paddd           m1, m3
    paddd           m0, m2
    psrad           m1, 5
    psrad           m0, 5
    packusdw        m0, m1
    pmulhuw         m0, m10
    mova    [dstq+r10], m0
    add            r10, 32
    jl .hv_loop
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    mov             t1, t0
    mov             t0, t4
    add           dstq, dst_strideq
    ret
.v:
    mov            r10, wq
.v_loop:
    mova            m0, [t1+r10]
    paddw           m2, m0, [t3+r10]
    mova            m1, [t2+r10]
    mova            m4, [t4+r10]
    punpckhwd       m3, m2, m1
    pmaddwd         m3, m14
    punpcklwd       m2, m1
    pmaddwd         m2, m14
    punpckhwd       m1, m0, m4
    pmaddwd         m1, m13
    punpcklwd       m0, m4
    pmaddwd         m0, m13
    paddd           m3, m9
    paddd           m2, m9
    paddd           m1, m3
    paddd           m0, m2
    psrad           m1, 5
    psrad           m0, 5
    packusdw        m0, m1
    pmulhuw         m0, m10
    mova    [dstq+r10], m0
    add            r10, 32
    jl .v_loop
    ret

%endif ; ARCH_X86_64
