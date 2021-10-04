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

wiener_shufA:  db  1,  2,  7,  6,  3,  4,  9,  8,  5,  6, 11, 10,  7,  8, 13, 12
wiener_shufB:  db  2,  3,  8,  7,  4,  5, 10,  9,  6,  7, 12, 11,  8,  9, 14, 13
wiener_shufC:  db  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11
wiener_shufD:  db  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12
wiener_perm32: db  1,  9,  3, 11,  5, 13,  7, 15, 33, 41, 35, 43, 37, 45, 39, 47
               db 17, 25, 19, 27, 21, 29, 23, 31, 49, 57, 51, 59, 53, 61, 55, 63
r_ext_mask:    times 68 db -1
               times  4 db  0
wiener_x_shuf: db  0,  2, -1,  0
wiener_x_add:  db  0,  1,127,  0

pd_m16380:     dd -16380
pd_8421376:    dd 8421376

SECTION .text

DECLARE_REG_TMP 4, 9, 7, 11, 12, 13, 14 ; ring buffer pointers

INIT_ZMM avx512icl
cglobal wiener_filter7_8bpc, 5, 15, 20, -384*12-16, dst, dst_stride, left, lpf, \
                                                    lpf_stride, w, edge, flt, h
    mov           fltq, fltmp
    mov          edged, r8m
    mov             wd, wm
    mov             hd, r6m
    vbroadcasti32x4 m6, [wiener_shufA]
    vbroadcasti32x4 m7, [wiener_shufB]
    mov           r10d, 0xfffe
    vbroadcasti32x4 m8, [wiener_shufC]
    vbroadcasti32x4 m9, [wiener_shufD]
    kmovw           k1, r10d
    vpbroadcastd    m0, [wiener_x_shuf]
    vpbroadcastd    m1, [wiener_x_add]
    mov            r10, 0xaaaaaaaaaaaaaaaa
    vpbroadcastd   m11, [fltq+ 0]
    vpbroadcastd   m12, [fltq+ 4]
    kmovq           k2, r10
    vpbroadcastd   m10, [pd_m16380]
    packsswb       m11, m11 ; x0   x1   x0   x1
    vpbroadcastd   m14, [fltq+16]
    pshufb         m12, m0
    vpbroadcastd   m15, [fltq+20]
    paddb          m12, m1  ; x2   x3+1 x2   127
    vpbroadcastd   m13, [pd_8421376]
    psllw          m14, 5   ; y0 y1
    psllw          m15, 5   ; y2 y3
    cmp             wd, 32  ; the minimum lr unit size for chroma in 4:2:0 is 32
    jle .w32                ; pixels, so we need a special case for small widths
    lea             t1, [rsp+wq*2+16]
    add           lpfq, wq
    add           dstq, wq
    neg             wq
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
.h:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
    movd          xm16, [leftq]
    vmovdqu32  m16{k1}, [lpfq+r10-4]
    add          leftq, 4
    jmp .h_main
.h_extend_left:
    vpbroadcastb  xm16, [lpfq+r10]   ; the masked load ensures that no exception
    vmovdqu32  m16{k1}, [lpfq+r10-4] ; gets raised from accessing invalid memory
    jmp .h_main
.h_top:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .h_extend_left
.h_loop:
    movu           m16, [lpfq+r10-4]
.h_main:
    movu           m17, [lpfq+r10+4]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .h_have_right
    cmp           r10d, -66
    jl .h_have_right
    push            r0
    lea             r0, [r_ext_mask+65]
    vpbroadcastb    m0, [lpfq-1]
    vpternlogd     m16, m0, [r0+r10+0], 0xe4 ; c ? a : b
    vpternlogd     m17, m0, [r0+r10+8], 0xe4
    pop             r0
.h_have_right:
    pshufb          m4, m16, m6
    mova            m0, m10
    vpdpbusd        m0, m4, m11
    pshufb          m4, m16, m7
    mova            m2, m10
    vpdpbusd        m2, m4, m11
    pshufb          m4, m17, m6
    mova            m1, m10
    vpdpbusd        m1, m4, m11
    pshufb          m4, m17, m7
    mova            m3, m10
    vpdpbusd        m3, m4, m11
    pshufb          m4, m16, m8
    vpdpbusd        m0, m4, m12
    pshufb         m16, m9
    vpdpbusd        m2, m16, m12
    pshufb          m4, m17, m8
    vpdpbusd        m1, m4, m12
    pshufb         m17, m9
    vpdpbusd        m3, m17, m12
    packssdw        m0, m2
    packssdw        m1, m3
    psraw           m0, 3
    psraw           m1, 3
    mova [t1+r10*2+ 0], m0
    mova [t1+r10*2+64], m1
    add            r10, 64
    jl .h_loop
    ret
ALIGN function_align
.hv:
    add           lpfq, dst_strideq
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
    movd          xm16, [leftq]
    vmovdqu32  m16{k1}, [lpfq+r10-4]
    add          leftq, 4
    jmp .hv_main
.hv_extend_left:
    vpbroadcastb  xm16, [lpfq+r10]
    vmovdqu32  m16{k1}, [lpfq+r10-4]
    jmp .hv_main
.hv_bottom:
    mov            r10, wq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .hv_extend_left
.hv_loop:
    movu           m16, [lpfq+r10-4]
.hv_main:
    movu           m17, [lpfq+r10+4]
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .hv_have_right
    cmp           r10d, -66
    jl .hv_have_right
    push            r0
    lea             r0, [r_ext_mask+65]
    vpbroadcastb    m0, [lpfq-1]
    vpternlogd     m16, m0, [r0+r10+0], 0xe4 ; c ? a : b
    vpternlogd     m17, m0, [r0+r10+8], 0xe4
    pop             r0
.hv_have_right:
    pshufb          m4, m16, m6
    mova            m0, m10
    vpdpbusd        m0, m4, m11
    pshufb          m4, m16, m7
    mova            m2, m10
    vpdpbusd        m2, m4, m11
    pshufb          m4, m17, m6
    mova            m1, m10
    vpdpbusd        m1, m4, m11
    pshufb          m4, m17, m7
    mova            m3, m10
    vpdpbusd        m3, m4, m11
    pshufb          m4, m16, m8
    vpdpbusd        m0, m4, m12
    pshufb         m16, m9
    vpdpbusd        m2, m16, m12
    pshufb          m4, m17, m8
    vpdpbusd        m1, m4, m12
    pshufb         m17, m9
    vpdpbusd        m3, m17, m12
    packssdw        m0, m2
    packssdw        m1, m3
    psraw           m0, 3
    psraw           m1, 3
    mova           m16, [t4+r10*2]
    paddw          m16, [t2+r10*2]
    mova            m3, [t3+r10*2]
    mova           m17, [t4+r10*2+64]
    paddw          m17, [t2+r10*2+64]
    mova            m5, [t3+r10*2+64]
    punpcklwd       m4, m16, m3
    mova            m2, m13
    vpdpwssd        m2, m4, m15
    punpcklwd      m18, m17, m5
    mova            m4, m13
    vpdpwssd        m4, m18, m15
    punpckhwd      m16, m3
    mova            m3, m13
    vpdpwssd        m3, m16, m15
    punpckhwd      m17, m5
    mova            m5, m13
    vpdpwssd        m5, m17, m15
    mova           m17, [t5+r10*2]
    paddw          m17, [t1+r10*2]
    paddw          m16, m0, [t6+r10*2]
    mova           m19, [t5+r10*2+64]
    paddw          m19, [t1+r10*2+64]
    paddw          m18, m1, [t6+r10*2+64]
    mova [t0+r10*2+ 0], m0
    mova [t0+r10*2+64], m1
    punpcklwd       m0, m16, m17
    vpdpwssd        m2, m0, m14
    punpcklwd       m1, m18, m19
    vpdpwssd        m4, m1, m14
    punpckhwd      m16, m17
    vpdpwssd        m3, m16, m14
    punpckhwd      m18, m19
    vpdpwssd        m5, m18, m14
    packuswb        m2, m4
    psrlw           m2, 8
    vpackuswb   m2{k2}, m3, m5
    mova    [dstq+r10], m2
    add            r10, 64
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
    mova            m4, [t4+r10*2+ 0]
    paddw           m4, [t2+r10*2+ 0]
    mova            m1, [t3+r10*2+ 0]
    mova            m5, [t4+r10*2+64]
    paddw           m5, [t2+r10*2+64]
    mova            m3, [t3+r10*2+64]
    punpcklwd       m6, m4, m1
    mova            m0, m13
    vpdpwssd        m0, m6, m15
    punpcklwd       m6, m5, m3
    mova            m2, m13
    vpdpwssd        m2, m6, m15
    punpckhwd       m4, m1
    mova            m1, m13
    vpdpwssd        m1, m4, m15
    punpckhwd       m5, m3
    mova            m3, m13
    vpdpwssd        m3, m5, m15
    mova            m5, [t1+r10*2+ 0]
    paddw           m4, m5, [t6+r10*2+ 0]
    paddw           m5, [t5+r10*2+ 0]
    mova            m7, [t1+r10*2+64]
    paddw           m6, m7, [t6+r10*2+64]
    paddw           m7, [t5+r10*2+64]
    punpcklwd       m8, m4, m5
    vpdpwssd        m0, m8, m14
    punpcklwd       m8, m6, m7
    vpdpwssd        m2, m8, m14
    punpckhwd       m4, m5
    vpdpwssd        m1, m4, m14
    punpckhwd       m6, m7
    vpdpwssd        m3, m6, m14
    packuswb        m0, m2
    psrlw           m0, 8
    vpackuswb   m0{k2}, m1, m3
    mova    [dstq+r10], m0
    add            r10, 64
    jl .v_loop
    mov             t6, t5
    mov             t5, t4
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    add           dstq, dst_strideq
    ret
.w32:
    lea            r10, [r_ext_mask+73]
    mova          ym18, [wiener_perm32]
    lea             t1, [rsp+16]
    sub            r10, wq
    test         edgeb, 4 ; LR_HAVE_TOP
    jz .w32_no_top
    call .w32_h_top
    add           lpfq, lpf_strideq
    mov             t6, t1
    mov             t5, t1
    add             t1, 32*2
    call .w32_h_top
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov             t4, t1
    add             t1, 32*2
    mov      [rsp+8*1], lpf_strideq
    add             r7, lpf_strideq
    mov      [rsp+8*0], r7 ; below
    call .w32_h
    mov             t3, t1
    mov             t2, t1
    dec             hd
    jz .w32_v1
    add           lpfq, dst_strideq
    add             t1, 32*2
    call .w32_h
    mov             t2, t1
    dec             hd
    jz .w32_v2
    add           lpfq, dst_strideq
    add             t1, 32*2
    call .w32_h
    dec             hd
    jz .w32_v3
.w32_main:
    lea             t0, [t1+32*2]
.w32_main_loop:
    call .w32_hv
    dec             hd
    jnz .w32_main_loop
    test         edgeb, 8 ; LR_HAVE_BOTTOM
    jz .w32_v3
    mov           lpfq, [rsp+8*0]
    call .w32_hv_bottom
    add           lpfq, [rsp+8*1]
    call .w32_hv_bottom
.w32_v1:
    call .w32_v
    RET
.w32_no_top:
    lea             r7, [lpfq+lpf_strideq*4]
    mov           lpfq, dstq
    mov      [rsp+8*1], lpf_strideq
    lea             r7, [r7+lpf_strideq*2]
    mov      [rsp+8*0], r7
    call .w32_h
    mov             t6, t1
    mov             t5, t1
    mov             t4, t1
    mov             t3, t1
    mov             t2, t1
    dec             hd
    jz .w32_v1
    add           lpfq, dst_strideq
    add             t1, 32*2
    call .w32_h
    mov             t2, t1
    dec             hd
    jz .w32_v2
    add           lpfq, dst_strideq
    add             t1, 32*2
    call .w32_h
    dec             hd
    jz .w32_v3
    lea             t0, [t1+32*2]
    call .w32_hv
    dec             hd
    jz .w32_v3
    add             t0, 32*8
    call .w32_hv
    dec             hd
    jnz .w32_main
.w32_v3:
    call .w32_v
.w32_v2:
    call .w32_v
    jmp .w32_v1
.w32_h:
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .w32_h_extend_left
    movd          xm16, [leftq]
    vmovdqu32 ym16{k1}, [lpfq-4]
    add          leftq, 4
    jmp .w32_h_main
.w32_h_extend_left:
    vpbroadcastb  xm16, [lpfq]   ; the masked load ensures that no exception
    vmovdqu32 ym16{k1}, [lpfq-4] ; gets raised from accessing invalid memory
    jmp .w32_h_main
.w32_h_top:
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .w32_h_extend_left
    movu          ym16, [lpfq-4]
.w32_h_main:
    vinserti32x8   m16, [lpfq+4], 1
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .w32_h_have_right
    vpbroadcastb    m0, [lpfq+wq-1]
    movu          ym17, [r10-8]
    vinserti32x8   m17, [r10+0], 1
    vpternlogd     m16, m0, m17, 0xe4 ; c ? a : b
.w32_h_have_right:
    pshufb          m2, m16, m6
    mova            m0, m10
    vpdpbusd        m0, m2, m11
    pshufb          m2, m16, m7
    mova            m1, m10
    vpdpbusd        m1, m2, m11
    pshufb          m2, m16, m8
    vpdpbusd        m0, m2, m12
    pshufb         m16, m9
    vpdpbusd        m1, m16, m12
    packssdw        m0, m1
    psraw           m0, 3
    mova          [t1], m0
    ret
.w32_hv:
    add           lpfq, dst_strideq
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .w32_hv_extend_left
    movd          xm16, [leftq]
    vmovdqu32 ym16{k1}, [lpfq-4]
    add          leftq, 4
    jmp .w32_hv_main
.w32_hv_extend_left:
    vpbroadcastb  xm16, [lpfq]
    vmovdqu32 ym16{k1}, [lpfq-4]
    jmp .w32_hv_main
.w32_hv_bottom:
    test         edgeb, 1 ; LR_HAVE_LEFT
    jz .w32_hv_extend_left
    movu          ym16, [lpfq-4]
.w32_hv_main:
    vinserti32x8   m16, [lpfq+4], 1
    test         edgeb, 2 ; LR_HAVE_RIGHT
    jnz .w32_hv_have_right
    vpbroadcastb    m0, [lpfq+wq-1]
    movu          ym17, [r10-8]
    vinserti32x8   m17, [r10+0], 1
    vpternlogd     m16, m0, m17, 0xe4
.w32_hv_have_right:
    mova            m3, [t4]
    paddw           m3, [t2]
    mova            m2, [t3]
    pshufb          m4, m16, m6
    mova            m0, m10
    vpdpbusd        m0, m4, m11
    pshufb          m4, m16, m7
    mova            m5, m10
    vpdpbusd        m5, m4, m11
    punpcklwd       m4, m3, m2
    mova            m1, m13
    vpdpwssd        m1, m4, m15
    punpckhwd       m3, m2
    mova            m2, m13
    vpdpwssd        m2, m3, m15
    pshufb          m4, m16, m8
    vpdpbusd        m0, m4, m12
    pshufb         m16, m9
    vpdpbusd        m5, m16, m12
    packssdw        m0, m5
    psraw           m0, 3
    mova            m4, [t5]
    paddw           m4, [t1]
    paddw           m3, m0, [t6]
    mova          [t0], m0
    punpcklwd       m0, m3, m4
    vpdpwssd        m1, m0, m14
    punpckhwd       m3, m4
    vpdpwssd        m2, m3, m14
    packuswb        m1, m2
    vpermb         m16, m18, m1
    mova        [dstq], ym16
    mov             t6, t5
    mov             t5, t4
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    mov             t1, t0
    mov             t0, t6
    add           dstq, dst_strideq
    ret
.w32_v:
    mova            m2, [t4]
    paddw           m2, [t2]
    mova            m1, [t3]
    mova            m4, [t1]
    paddw           m3, m4, [t6]
    paddw           m4, [t5]
    punpcklwd       m5, m2, m1
    mova            m0, m13
    vpdpwssd        m0, m5, m15
    punpckhwd       m2, m1
    mova            m1, m13
    vpdpwssd        m1, m2, m15
    punpcklwd       m2, m3, m4
    vpdpwssd        m0, m2, m14
    punpckhwd       m3, m4
    vpdpwssd        m1, m3, m14
    packuswb        m0, m1
    vpermb         m16, m18, m0
    mova        [dstq], ym16
    mov             t6, t5
    mov             t5, t4
    mov             t4, t3
    mov             t3, t2
    mov             t2, t1
    add           dstq, dst_strideq
    ret

%endif ; ARCH_X86_64
