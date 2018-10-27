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
pd_04512763: dd 0, 4, 5, 1, 2, 7, 6, 3
div_table: dd 840, 420, 280, 210, 168, 140, 120, 105
           dd 420, 210, 140, 105
pd_04261537: dd 0, 4, 2, 6, 1, 5, 3, 7
shufw_6543210x: db 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15
shufw_210xxxxx: db 4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
pw_128: times 2 dw 128

SECTION .text

INIT_YMM avx2
cglobal cdef_dir, 3, 4, 15, src, stride, var, stride3
    lea       stride3q, [strideq*3]
    movq           xm0, [srcq+strideq*0]
    movq           xm1, [srcq+strideq*1]
    movq           xm2, [srcq+strideq*2]
    movq           xm3, [srcq+stride3q]
    lea           srcq, [srcq+strideq*4]
    vpbroadcastq    m4, [srcq+strideq*0]
    vpbroadcastq    m5, [srcq+strideq*1]
    vpbroadcastq    m6, [srcq+strideq*2]
    vpbroadcastq    m7, [srcq+stride3q]
    vpbroadcastd    m8, [pw_128]
    pxor            m9, m9

    vpblendd        m0, m0, m7, 0xf0
    vpblendd        m1, m1, m6, 0xf0
    vpblendd        m2, m2, m5, 0xf0
    vpblendd        m3, m3, m4, 0xf0

    punpcklbw       m0, m9
    punpcklbw       m1, m9
    punpcklbw       m2, m9
    punpcklbw       m3, m9

    psubw           m0, m8
    psubw           m1, m8
    psubw           m2, m8
    psubw           m3, m8

    ; shuffle registers to generate partial_sum_diag[0-1] together
    vpermq          m7, m0, q1032
    vpermq          m6, m1, q1032
    vpermq          m5, m2, q1032
    vpermq          m4, m3, q1032

    ; start with partial_sum_hv[0-1]
    paddw           m8, m0, m1
    paddw           m9, m2, m3
    phaddw         m10, m0, m1
    phaddw         m11, m2, m3
    paddw           m8, m9
    phaddw         m10, m11
    vextracti128   xm9, m8, 1
    vextracti128  xm11, m10, 1
    paddw          xm8, xm9                 ; partial_sum_hv[1]
    phaddw        xm10, xm11                ; partial_sum_hv[0]
    vinserti128     m8, xm10, 1
    vpbroadcastd    m9, [div_table+44]
    pmaddwd         m8, m8
    pmulld          m8, m9                  ; cost6[2a-d] | cost2[a-d]

    ; create aggregates [lower half]:
    ; m9 = m0:01234567+m1:x0123456+m2:xx012345+m3:xxx01234+
    ;      m4:xxxx0123+m5:xxxxx012+m6:xxxxxx01+m7:xxxxxxx0
    ; m10=             m1:7xxxxxxx+m2:67xxxxxx+m3:567xxxxx+
    ;      m4:4567xxxx+m5:34567xxx+m6:234567xx+m7:1234567x
    ; and [upper half]:
    ; m9 = m0:xxxxxxx0+m1:xxxxxx01+m2:xxxxx012+m3:xxxx0123+
    ;      m4:xxx01234+m5:xx012345+m6:x0123456+m7:01234567
    ; m10= m0:1234567x+m1:234567xx+m2:34567xxx+m3:4567xxxx+
    ;      m4:567xxxxx+m5:67xxxxxx+m6:7xxxxxxx
    ; and then shuffle m11 [shufw_6543210x], unpcklwd, pmaddwd, pmulld, paddd

    pslldq          m9, m1, 2
    psrldq         m10, m1, 14
    pslldq         m11, m2, 4
    psrldq         m12, m2, 12
    pslldq         m13, m3, 6
    psrldq         m14, m3, 10
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14
    pslldq         m11, m4, 8
    psrldq         m12, m4, 8
    pslldq         m13, m5, 10
    psrldq         m14, m5, 6
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14
    pslldq         m11, m6, 12
    psrldq         m12, m6, 4
    pslldq         m13, m7, 14
    psrldq         m14, m7, 2
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14                 ; partial_sum_diag[0/1][8-14,zero]
    vbroadcasti128 m14, [shufw_6543210x]
    vbroadcasti128 m13, [div_table+16]
    vbroadcasti128 m12, [div_table+0]
    paddw           m9, m0                  ; partial_sum_diag[0/1][0-7]
    pshufb         m10, m14
    punpckhwd      m11, m9, m10
    punpcklwd       m9, m10
    pmaddwd        m11, m11
    pmaddwd         m9, m9
    pmulld         m11, m13
    pmulld          m9, m12
    paddd           m9, m11                 ; cost0[a-d] | cost4[a-d]

    ; merge horizontally and vertically for partial_sum_alt[0-3]
    paddw          m10, m0, m1
    paddw          m11, m2, m3
    paddw          m12, m4, m5
    paddw          m13, m6, m7
    phaddw          m0, m4
    phaddw          m1, m5
    phaddw          m2, m6
    phaddw          m3, m7

    ; create aggregates [lower half]:
    ; m4 = m10:01234567+m11:x0123456+m12:xx012345+m13:xxx01234
    ; m11=              m11:7xxxxxxx+m12:67xxxxxx+m13:567xxxxx
    ; and [upper half]:
    ; m4 = m10:xxx01234+m11:xx012345+m12:x0123456+m13:01234567
    ; m11= m10:567xxxxx+m11:67xxxxxx+m12:7xxxxxxx
    ; and then shuffle m11 [shufw_210xxxxx], unpcklwd, pmaddwd, pmulld, paddd

    vbroadcasti128 m14, [shufw_210xxxxx]
    pslldq          m4, m11, 2
    psrldq         m11, 14
    pslldq          m5, m12, 4
    psrldq         m12, 12
    pslldq          m6, m13, 6
    psrldq         m13, 10
    paddw           m4, m10
    paddw          m11, m12
    vpbroadcastd   m12, [div_table+44]
    paddw           m5, m6
    paddw          m11, m13                 ; partial_sum_alt[3/2] right
    vbroadcasti128 m13, [div_table+32]
    paddw           m4, m5                  ; partial_sum_alt[3/2] left
    pshufb         m11, m14
    punpckhwd       m6, m4, m11
    punpcklwd       m4, m11
    pmaddwd         m6, m6
    pmaddwd         m4, m4
    pmulld          m6, m12
    pmulld          m4, m13
    paddd           m4, m6                  ; cost7[a-d] | cost5[a-d]

    ; create aggregates [lower half]:
    ; m5 = m0:01234567+m1:x0123456+m2:xx012345+m3:xxx01234
    ; m1 =             m1:7xxxxxxx+m2:67xxxxxx+m3:567xxxxx
    ; and [upper half]:
    ; m5 = m0:xxx01234+m1:xx012345+m2:x0123456+m3:01234567
    ; m1 = m0:567xxxxx+m1:67xxxxxx+m2:7xxxxxxx
    ; and then shuffle m11 [shufw_210xxxxx], unpcklwd, pmaddwd, pmulld, paddd

    pslldq          m5, m1, 2
    psrldq          m1, 14
    pslldq          m6, m2, 4
    psrldq          m2, 12
    pslldq          m7, m3, 6
    psrldq          m3, 10
    paddw           m5, m0
    paddw           m1, m2
    paddw           m6, m7
    paddw           m1, m3                  ; partial_sum_alt[0/1] right
    paddw           m5, m6                  ; partial_sum_alt[0/1] left
    pshufb          m1, m14
    punpckhwd       m6, m5, m1
    punpcklwd       m5, m1
    pmaddwd         m6, m6
    pmaddwd         m5, m5
    pmulld          m6, m12
    pmulld          m5, m13
    paddd           m5, m6                  ; cost1[a-d] | cost3[a-d]

    mova           xm0, [pd_04512763+ 0]
    mova           xm1, [pd_04512763+ 16]
    phaddd          m9, m8
    phaddd          m5, m4
    phaddd          m9, m5
    vpermd          m0, m9                  ; cost[0/4/2/6]
    vpermd          m1, m9                  ; cost[1/5/3/7]

    ; now find the best cost, its idx^4 complement, and its idx
    pcmpgtd        xm2, xm1, xm0            ; [1/5/3/7] > [0/4/2/6]
    pand           xm3, xm2, xm1
    pandn          xm4, xm2, xm0
    por            xm3, xm4                 ; higher 4 values
    pshufd         xm1, xm1, q2301
    pshufd         xm0, xm0, q2301
    pand           xm1, xm2, xm1
    pandn          xm0, xm2, xm0
    por            xm0, xm1                 ; complementary 4 values at idx^4 offset
    pand          xm13, xm2, [pd_04261537+16]
    pandn         xm14, xm2, [pd_04261537+ 0]
    por           xm14, xm13                ; indices

    punpckhqdq     xm4, xm3, xm0
    punpcklqdq     xm3, xm0
    pcmpgtd        xm5, xm4, xm3            ; [2or3-6or7] > [0or1/4or5]
    punpcklqdq     xm5, xm5
    pand           xm6, xm5, xm4
    pandn          xm7, xm5, xm3
    por            xm6, xm7                 ; { highest 2 values, complements at idx^4 }
    movhlps       xm13, xm14
    pand          xm13, xm5, xm13
    pandn         xm14, xm5, xm14
    por           xm14, xm13

    pshufd         xm7, xm6, q3311
    pcmpgtd        xm8, xm7, xm6            ; [4or5or6or7] > [0or1or2or3]
    punpcklqdq     xm8, xm8
    pand           xm9, xm8, xm7
    pandn         xm10, xm8, xm6
    por            xm9, xm10                ; max
    movhlps       xm10, xm9                 ; complement at idx^4
    psubd          xm9, xm10
    psrld          xm9, 10
    movd        [varq], xm9
    pshufd        xm13, xm14, q1111
    pand          xm13, xm8, xm13
    pandn         xm14, xm8, xm14
    por           xm14, xm13
    movd           eax, xm14
    RET
%endif ; ARCH_X86_64
