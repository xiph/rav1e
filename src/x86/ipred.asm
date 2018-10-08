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

paeth_shuf:  db  7,  7,  7,  7,  3,  3,  3,  3,  6,  6,  6,  6,  2,  2,  2,  2
             db  5,  5,  5,  5,  1,  1,  1,  1,  4,  4,  4,  4,  0,  0,  0,  0

pb_1:   times 4 db 1
pb_128: times 4 db 128

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

JMP_TABLE ipred_paeth,    avx2, w4, w8, w16, w32, w64
JMP_TABLE ipred_dc,       avx2, h4, h8, h16, h32, h64, w4, w8, w16, w32, w64, \
                                s4-10*4, s8-10*4, s16-10*4, s32-10*4, s64-10*4
JMP_TABLE ipred_dc_left,  avx2, h4, h8, h16, h32, h64
JMP_TABLE ipred_h,        avx2, w4, w8, w16, w32, w64

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
    lea                  r5, [ipred_paeth_avx2_table]
    tzcnt                wd, wm
    vpbroadcastb         m5, [tlq]   ; topleft
    movifnidn            hd, hm
    movsxd               wq, [r5+wq*4]
    vpbroadcastd         m4, [r5-ipred_paeth_avx2_table+pb_1]
    add                  wq, r5
    jmp                  wq
.w4:
    vpbroadcastd         m6, [tlq+1] ; top
    mova                 m8, [r5-ipred_paeth_avx2_table+paeth_shuf]
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
    pextrd [dstq+strideq*1], xm0, 2
    movd   [dstq+strideq*2], xm1
    pextrd [dstq+r3       ], xm1, 2
    cmp                  hd, 4
    je .ret
    lea                dstq, [dstq+strideq*4]
    pextrd [dstq+strideq*0], xm0, 1
    pextrd [dstq+strideq*1], xm0, 3
    pextrd [dstq+strideq*2], xm1, 1
    pextrd [dstq+r3       ], xm1, 3
    lea                dstq, [dstq+strideq*4]
    sub                  hd, 8
    jg .w4_loop
.ret:
    RET
ALIGN function_align
.w8:
    vpbroadcastq         m6, [tlq+1]
    mova                 m8, [r5-ipred_paeth_avx2_table+paeth_shuf]
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
    movhps [dstq+strideq*1], xm0
    movq   [dstq+strideq*2], xm1
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

%endif
