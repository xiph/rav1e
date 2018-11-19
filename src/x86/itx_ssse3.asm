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


SECTION_RODATA 16

deint_shuf: db  0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15

qw_2896x8:      times 8 dw  2896*8
qw_1567_m3784:  times 4 dw  1567, -3784
qw_3784_1567:   times 4 dw  3784,  1567

pd_2048:        times 4 dd  2048
qw_2048:        times 8 dw  2048

SECTION .text

%define m(x) mangle(private_prefix %+ _ %+ x %+ SUFFIX)

%macro ITX4_END 4-5 2048 ; row[1-4], rnd
%if %5
    mova                 m2, [qw_%5]
    pmulhrsw             m0, m2
    pmulhrsw             m1, m2
%endif
    lea                  r2, [dstq+strideq*2]
%assign %%i 1
%rep 4
    %if %1 & 2
        CAT_XDEFINE %%row_adr, %%i, r2   + strideq*(%1&1)
    %else
        CAT_XDEFINE %%row_adr, %%i, dstq + strideq*(%1&1)
    %endif
    %assign %%i %%i + 1
    %rotate 1
%endrep

    movd                 m2, [%%row_adr1]       ;dst0
    movd                 m4, [%%row_adr2]       ;dst1
    punpckldq            m2, m4                 ;high: dst1 :low: dst0
    movd                 m3, [%%row_adr3]       ;dst2
    movd                 m4, [%%row_adr4]       ;dst3
    punpckldq            m3, m4                 ;high: dst3 :low: dst2

    pxor                 m4, m4
    punpcklbw            m2, m4                 ;extend byte to word
    punpcklbw            m3, m4                 ;extend byte to word

    paddw                m0, m2                 ;high: dst1 + out1 ;low: dst0 + out0
    paddw                m1, m3                 ;high: dst3 + out3 ;low: dst2 + out2

    packuswb             m0, m1                 ;high->low: dst3 + out3, dst2 + out2, dst1 + out1, dst0 + out0

    movd       [%%row_adr1], m0                 ;store dst0 + out0
    pshuflw              m1, m0, q1032
    movd       [%%row_adr2], m1                 ;store dst1 + out1
    punpckhqdq           m0, m0
    movd       [%%row_adr3], m0                 ;store dst2 + out2
    psrlq                m0, 32
    movd       [%%row_adr4], m0                 ;store dst3 + out3

  RET
%endmacro


; flags: 1 = swap, 2: coef_regs
%macro ITX_MUL2X_PACK 5-6 0 ; dst/src, tmp[1], rnd, coef[1-2], flags
%if %6 & 2
    pmaddwd              m%2, m%4, m%1
    pmaddwd              m%1, m%5
%elif %6 & 1
    pmaddwd              m%2, m%1, [qw_%5_%4]
    pmaddwd              m%1, [qw_%4_m%5]
%else
    pmaddwd              m%2, m%1, [qw_%4_m%5]
    pmaddwd              m%1, [qw_%5_%4]
%endif
    paddd                m%2, m%3
    paddd                m%1, m%3
    psrad                m%2, 12
    psrad                m%1, 12
    packssdw             m%1, m%2
%endmacro

%macro IDCT4_1D_PACKED 0-1   ;qw_2896x8
    punpckhwd            m2, m0, m1           ;unpacked in1 in3
    psubw                m3, m0, m1
    paddw                m0, m1
    punpcklqdq           m0, m3               ;high: in0-in2 ;low: in0+in2

    mova                 m3, [pd_2048]
    ITX_MUL2X_PACK 2, 1, 3, 1567, 3784

%if %0 == 1
    pmulhrsw             m0, m%1
%else
    pmulhrsw             m0, [qw_2896x8]     ;high: t1 ;low: t0
%endif

    psubw                m1, m0, m2          ;high: out2 ;low: out3
    paddw                m0, m2              ;high: out1 ;low: out0
%endmacro


%macro INV_TXFM_FN 4 ; type1, type2, fast_thresh, size
cglobal inv_txfm_add_%1_%2_%4, 4, 5, 0, dst, stride, coeff, eob, tx2
    %undef cmp
    lea tx2q, [m(i%2_%4_internal).pass2]
%if %3 > 0
    cmp                  eobd, %3
    jle %%end
%elif %3 == 0
    test                 eobd, eobd
    jz %%end
%endif
    call i%1_%4_internal
    RET
ALIGN function_align
%%end:
%endmacro

%macro INV_TXFM_4X4_FN 2-3 -1 ; type1, type2, fast_thresh
    INV_TXFM_FN          %1, %2, %3, 4x4
%ifidn %1_%2, dct_identity
%elifidn %1_%2, identity_dct
%elif %3 >= 0
    pshuflw              m0, [coeffq], q0000
    punpcklqdq           m0, m0
%ifidn %1, dct
    mova                 m1, [qw_2896x8]
    pmulhrsw             m0, m1
%elifidn %1, adst
%elifidn %1, flipadst
%endif
    mov            [coeffq], eobd             ;0
%ifidn %2, dct
%ifnidn %1, dct
    pmulhrsw             m0, [qw_2896x8]
%else
    pmulhrsw             m0, m1
%endif
    mova                 m1, m0
    ITX4_END             0, 1, 2, 3
%else ; adst / flipadst
%endif
%endif
%endmacro


INIT_XMM ssse3

cglobal idct_4x4_internal, 0, 0, 4, dst, stride, coeff, eob, tx2
    mova                 m0, [coeffq+16*0]      ;high: in1 ;low: in0
    mova                 m1, [coeffq+16*1]      ;high: in3 ;low in2

    IDCT4_1D_PACKED

    mova                 m2, [deint_shuf]
    shufps               m3, m0, m1, q1331
    shufps               m0, m1, q0220
    pshufb               m0, m2                 ;high: in1 ;low: in0
    pshufb               m1, m3, m2             ;high: in3 ;low :in2
    jmp                tx2q

.pass2:
    IDCT4_1D_PACKED

    pxor                 m2, m2
    mova      [coeffq+16*0], m2
    mova      [coeffq+16*1], m2                 ;memset(coeff, 0, sizeof(*coeff) * sh * sw);

    ITX4_END     0, 1, 3, 2

INV_TXFM_4X4_FN dct, dct, 0
