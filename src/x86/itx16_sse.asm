; Copyright © 2021, VideoLAN and dav1d authors
; Copyright © 2021, Two Orioles, LLC
; Copyright © 2017-2021, The rav1e contributors
; Copyright © 2020, Nathan Egge
; Copyright © 2021, Matthias Dressel
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
%macro COEF 1
pd_%1: times 4 dd %1
%endmacro

COEF  201
COEF  401
COEF  601
COEF  799
COEF  995
COEF 1189
COEF 1380
COEF 1567
COEF 1751
COEF 1931
COEF 2106
COEF 2276
COEF 2440
COEF 2598
COEF 2751
COEF 2896
COEF 3035
COEF 3166
COEF 3290
COEF 3406
COEF 3513
COEF 3612
COEF 3703
COEF 3784
COEF 3857
COEF 3920
COEF 3973
COEF 4017
COEF 4052
COEF 4076
COEF 4091

deint_shuf:  db  0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15

pd_1321:         times 4 dd  1321
pd_2482:         times 4 dd  2482
pd_m3344:        times 4 dd -3344
pd_2048:         times 4 dd  2048
pw_2048:         times 8 dw  2048
pd_3803:         times 4 dd  3803
pd_5793:         times 4 dd  5793
pw_1697x8:       times 8 dw  1697*8
pw_2896x8:       times 8 dw  2896*8
pixel_10bpc_max: times 8 dw  0x03ff

pw_1567_3784:    times 4 dw  1567,  3784
pw_m3784_1567:   times 4 dw -3784,  1567

cextern inv_txfm_add_dct_dct_4x4_8bpc_ssse3
cextern iadst_4x4_internal_8bpc_ssse3.main

SECTION .text

%macro REPX 2-*
    %xdefine %%f(x) %1
%rep %0 - 1
    %rotate 1
    %%f(%1)
%endrep
%endmacro

%define m_suffix(x, sfx) mangle(private_prefix %+ _ %+ x %+ sfx)
%define m(x) m_suffix(x, SUFFIX)

; This refers to the first function in itx_sse i.e. the start of the text section
; which is needed as a base pointer for constants.
%define itx8_start m_suffix(inv_txfm_add_dct_dct_4x4_8bpc, _ssse3)

%if ARCH_X86_64
%define o(x) x
%else
%define o(x) r6-$$+x ; PIC
%endif

%macro IWHT4_1D 0
    ; m0 = in0,  m1 = in1,  m2 = in2,  m3 = in3
    paddd                m0, m1      ; in0 += in1
    psubd                m4, m2, m3  ; tmp0 = in2 - in3
    psubd                m5, m0, m4  ; tmp1 = (in0 - tmp0) >> 1
    psrad                m5, 1
    psubd                m2, m5, m1  ; in2 = tmp1 - in1
    psubd                m5, m3      ; in1 = tmp1 - in3
    psubd                m0, m5      ; in0 -= in1
    paddd                m4, m2      ; in3 = tmp0 + in2
    ; m0 = out0,  m1 = in1,  m2 = out2,  m3 = in3
    ; m4 = out3,  m5 = out1
%endmacro

INIT_XMM sse2
cglobal inv_txfm_add_wht_wht_4x4_16bpc, 3, 3, 6, dst, stride, c, eob, bdmax
    mova                 m0, [cq+16*0]
    mova                 m1, [cq+16*1]
    mova                 m2, [cq+16*2]
    mova                 m3, [cq+16*3]
    REPX       {psrad x, 2}, m0, m1, m2, m3
    IWHT4_1D
    punpckldq            m1, m0, m5
    punpckhdq            m3, m0, m5
    punpckldq            m5, m2, m4
    punpckhdq            m2, m4
    punpcklqdq           m0, m1, m5
    punpckhqdq           m1, m5
    punpcklqdq           m4, m3, m2
    punpckhqdq           m3, m2
    mova                 m2, m4
    IWHT4_1D
    packssdw             m0, m4 ; low: out3,  high: out0
    packssdw             m2, m5 ; low: out2,  high: out1
    pxor                 m4, m4
    mova          [cq+16*0], m4
    mova          [cq+16*1], m4
    mova          [cq+16*2], m4
    mova          [cq+16*3], m4
    lea                  r2, [dstq+strideq*2]
    movq                 m1, [dstq+strideq*0]
    movhps               m1, [r2  +strideq*1]
    movq                 m3, [r2  +strideq*0]
    movhps               m3, [dstq+strideq*1]
    movd                 m5, bdmaxm
    pshuflw              m5, m5, q0000  ; broadcast
    punpcklqdq           m5, m5         ; broadcast
    paddsw               m0, m1
    paddsw               m2, m3
    pmaxsw               m0, m4
    pmaxsw               m2, m4
    pminsw               m0, m5
    pminsw               m2, m5
    movhps [r2  +strideq*1], m0 ; write out0
    movhps [dstq+strideq*1], m2 ; write out1
    movq   [r2  +strideq*0], m2 ; write out2
    movq   [dstq+strideq*0], m0 ; write out3
    RET

; dst1 = (src1 * coef1 - src2 * coef2 + rnd) >> 12
; dst2 = (src1 * coef2 + src2 * coef1 + rnd) >> 12
; flags: 2 = inv_dst1, 4 = inv_dst2
; skip round/shift if rnd is not a number
%macro ITX_MULSUB_2D 8-9 0 ; dst/src[1-2], tmp[1-3], rnd, coef[1-2], flags
; %1 dst/src[1]
; %2 dst/src[2]
; %3 tmp[1]
; %4 tmp[2]
; %5 tmp[3]
; %6 rnd
; %7 coef[1]
; %8 coef[2]
; %9 flags
%ifnidn %7,%8   ; optimize when coef1 == coef2
%if %8 < 32
    pmulld              m%4, m%1, m%8
    pmulld              m%3, m%2, m%8
%else
    mova                m%3, [o(pd_%8)]
    pmulld              m%4, m%1, m%3
    pmulld              m%3, m%2
%endif
%endif
%if %7 < 32
    pmulld              m%1, m%7
    pmulld              m%2, m%7
%else
    mova                m%5, [o(pd_%7)]
    pmulld              m%1, m%5
    pmulld              m%2, m%5
%endif
%if %9 & 4  ; invert dst2
    paddd               m%4, m%2
    psubd               m%2, m%6, m%4
%else
%ifnum %6
%ifnidn %7,%8
    paddd               m%4, m%6
%else
    paddd               m%1, m%6
%endif
%endif
%ifnidn %7,%8
    paddd               m%2, m%4
%else
    mova                m%3, m%2
    paddd               m%2, m%1
%endif
%endif
%if %9 & 2  ; invert dst1
    psubd               m%3, m%1
    paddd               m%1, m%3, m%6
%else
%ifnum %6
%ifnidn %7,%8
    paddd               m%1, m%6
%endif
%endif
    psubd               m%1, m%3
%endif
%ifnum %6
    psrad               m%2, 12
    psrad               m%1, 12
%endif
%endmacro

%macro INV_TXFM_FN 4 ; type1, type2, eob_offset, size
cglobal inv_txfm_add_%1_%2_%4_16bpc, 4, 7, 8, dst, stride, c, eob, tx2
    %define %%p1 m(i%1_%4_internal_16bpc)
%if ARCH_X86_32
    LEA                  r6, $$
%ifidn %1_%2, dct_dct
    test               eobd, eobd
    jz %%end
%endif
    lea                tx2q, [o(m(i%2_%4_internal_16bpc).pass2)]
    call %%p1
    RET
%%end:
%else
    ; Jump to the 1st txfm function if we're not taking the fast path, which
    ; in turn performs an indirect jump to the 2nd txfm function.
    lea tx2q, [o(m(i%2_%4_internal_16bpc).pass2)]
%ifidn %1_%2, dct_dct
    test               eobd, eobd
    jnz %%p1
%else
%if %3
    add                eobd, %3
%endif
    ; jump to the 1st txfm function unless it's located directly after this
    times ((%%end - %%p1) >> 31) & 1 jmp %%p1
ALIGN function_align
%%end:
%endif
%endif
%endmacro

%macro INV_TXFM_4X4_FN 2 ; type1, type2
    INV_TXFM_FN          %1, %2, 0, 4x4
%ifidn %1_%2, dct_dct
    imul                r5d, [cq], 2896
    movd                 m1, [o(pw_2896x8)]
    mov                [cq], eobd ; 0
    add                 r5d, 2048
    sar                 r5d, 12
    movd                 m0, r5d
    packssdw             m0, m0
    pmulhrsw             m0, m1
    pshuflw              m0, m0, q0000
    punpcklqdq           m0, m0
    mova                 m1, m0
    jmp m(iadst_4x4_internal_16bpc).end
%endif
%endmacro

%macro IDCT4_1D 8 ; src[1-4], tmp[1-3], rnd
    ; butterfly rotation
    ITX_MULSUB_2D        %1, %3, %5, %6, %7, %8, 2896, 2896 ; %1 out1  %3 out0
    ITX_MULSUB_2D        %2, %4, %5, %6, %7, %8, 1567, 3784 ; %2 out2  %4 out3
    ; Hadamard rotation
    psubd               m%5, m%1, m%2
    paddd               m%2, m%1
    paddd               m%1, m%3, m%4
    psubd               m%3, m%4
    ; %1 (src1) = out0
    ; %2 (src2) = out1
    ; %3 (src3) = out3
    ; $5 (tmp1) = out2
%endmacro

INIT_XMM sse4

INV_TXFM_4X4_FN dct, dct
INV_TXFM_4X4_FN dct, identity
INV_TXFM_4X4_FN dct, adst
INV_TXFM_4X4_FN dct, flipadst

cglobal idct_4x4_internal_16bpc, 0, 7, 8, dst, stride, c, eob, tx2
    mova                 m0, [cq+16*0]
    mova                 m1, [cq+16*1]
    mova                 m2, [cq+16*2]
    mova                 m3, [cq+16*3]
    mova                 m5, [o(pd_2048)]
    IDCT4_1D              0, 1, 2, 3, 4, 6, 7, 5
    packssdw             m0, m1     ; out0 out1
    packssdw             m4, m2     ; out2 out3
    ; transpose
    punpckhwd            m2, m0, m4
    punpcklwd            m0, m4
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    ; m0 = out0 out1
    ; m1 = out2 out3
    ; m5 = pd_2048
    jmp                tx2q
.pass2:
    ; m0 = in0 in1
    ; m1 = in2 in3
    ; m5 = pd_2048
    mova                 m4, [o(pw_m3784_1567)]
    punpckhwd            m2, m1, m0
    psubw                m3, m0, m1
    paddw                m0, m1
    punpcklqdq           m0, m3
    pmaddwd              m4, m2
    pmaddwd              m2, [o(pw_1567_3784)]
    pmulhrsw             m0, [o(pw_2896x8)]     ; t0 t1
    paddd                m4, m5
    paddd                m2, m5
    psrad                m4, 12
    psrad                m2, 12
    packssdw             m2, m4     ; t3 t2
    psubsw               m1, m0, m2 ; tmp3 tmp2
    paddsw               m0, m2     ; tmp0 tmp1
    packssdw             m5, m5     ; pw_2048
    pmulhrsw             m0, m5
    pmulhrsw             m1, m5
    movq                 m2, [dstq+strideq*0]
    movhps               m2, [dstq+strideq*1]
    lea                  r5, [dstq+strideq*2]
    movq                 m3, [r5  +strideq*1]
    movhps               m3, [r5  +strideq*0]
    mova                 m5, [o(pixel_10bpc_max)]
    pxor                 m4, m4
    mova          [cq+16*0], m4
    mova          [cq+16*1], m4
    mova          [cq+16*2], m4
    mova          [cq+16*3], m4
    paddw                m0, m2
    paddw                m1, m3
    pmaxsw               m0, m4
    pmaxsw               m1, m4
    pminsw               m0, m5
    pminsw               m1, m5
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movhps [r5  +strideq*0], m1
    movq   [r5  +strideq*1], m1
    RET

INV_TXFM_4X4_FN adst, dct
INV_TXFM_4X4_FN adst, adst
INV_TXFM_4X4_FN adst, flipadst
INV_TXFM_4X4_FN adst, identity

cglobal iadst_4x4_internal_16bpc, 0, 7, 8, dst, stride, c, eob, tx2
    call .main
    ; transpose
    punpckhwd            m2, m0, m1
    punpcklwd            m0, m1
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    ; m0 = out0 out1
    ; m1 = out2 out3
    ; m5 = pd_2048
    jmp                tx2q
.pass2:
    ; m0 = in0 in1
    ; m1 = in2 in3
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_4x4_internal_8bpc, _ssse3).main
.end:
    mova                 m4, [o(pw_2048)]
    movq                 m2, [dstq+strideq*0]
    movhps               m2, [dstq+strideq*1]
    lea                  r5, [dstq+strideq*2]
    movq                 m3, [r5  +strideq*0]
    movhps               m3, [r5  +strideq*1]
    mova                 m5, [o(pixel_10bpc_max)]
    pmulhrsw             m0, m4
    pmulhrsw             m1, m4
    pxor                 m4, m4
    mova          [cq+16*0], m4
    mova          [cq+16*1], m4
    mova          [cq+16*2], m4
    mova          [cq+16*3], m4
    paddw                m0, m2
    paddw                m1, m3
    pmaxsw               m0, m4
    pmaxsw               m1, m4
    pminsw               m0, m5
    pminsw               m1, m5
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movq   [r5  +strideq*0], m1
    movhps [r5  +strideq*1], m1
    RET
ALIGN function_align
.main:
    mova                 m1, [cq+16*2]
    mova                 m3, [cq+16*3]
    mova                 m0, [o(pd_1321)]  ; SINPI_1_9
    mova                 m2, [o(pd_2482)]  ; SINPI_2_9
    mova                 m6, [o(pd_3803)]  ; SINPI_4_9
    mova                 m5, [cq+16*0]
    pmulld               m4, m0, m1        ; s[4] = SINPI_1_9 * T[2]
    pmulld               m7, m3, m6        ; s[6] = SINPI_4_9 * T[3]
    pmulld               m6, m1            ; s[3] = SINPI_4_9 * T[2]
    pmulld               m0, m5            ; s[0] = SINPI_1_9 * T[0]
    psubd                m1, m3            ; T[2] - T[3]
    pmulld               m3, m2            ; s[5] = SINPI_2_9 * T[3]
    pmulld               m2, m5            ; s[1] = SINPI_2_9 * T[0]
    paddd                m0, m6            ; s[0] += s[3]
    paddd                m0, m3            ; s[0] += s[5]
    mova                 m3, [o(pd_m3344)] ; -SINPI_3_9
    psubd                m2, m4            ; s[1] -= s[4]
    psubd                m2, m7            ; s[1] -= s[6]
    psubd                m1, m5            ; -b7 = (T[2] -T[3]) - T[0]
    pmulld               m1, m3            ; s[2]  = -SINPI_3_9 * -b7
    pmulld               m3, [cq+16*1]     ; -s[3] = -SINPI_3_9 * T[1]
    mova                 m5, [o(pd_2048)]
    REPX      {paddd x, m5}, m0, m1        ; {s[0], s[2]} + 2048
    paddd                m4, m0, m2        ; x[3]  = s[0] + s[1]
    psubd                m2, m3            ; x[1]  = s[1] + s[3]
    psubd                m0, m3            ; x[0]  = s[0] + s[3]
    paddd                m4, m3            ; x[3] -= s[3]
    paddd                m2, m5            ; x[1] + 2048
    REPX      {psrad x, 12}, m0, m2, m1, m4
    packssdw             m0, m2            ; out0 out1
    packssdw             m1, m4            ; out2 out3
    ret


INV_TXFM_4X4_FN flipadst, dct
INV_TXFM_4X4_FN flipadst, adst
INV_TXFM_4X4_FN flipadst, flipadst
INV_TXFM_4X4_FN flipadst, identity

cglobal iflipadst_4x4_internal_16bpc, 0, 7, 8, dst, stride, c, eob, tx2
    call m(iadst_4x4_internal_16bpc).main
    ; transpose
    punpcklwd            m2, m1, m0
    punpckhwd            m1, m0
    punpcklwd            m0, m1, m2
    punpckhwd            m1, m2
    ; m0 = out0 out1
    ; m1 = out2 out3
    ; m5 = pd_2048
    jmp                tx2q
.pass2:
    ; m0 = in0 in1
    ; m1 = in2 in3
%if ARCH_X86_32
    lea                 r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_4x4_internal_8bpc, _ssse3).main
    mova                 m4, [o(pw_2048)]
    movq                 m3, [dstq+strideq*1]
    movhps               m3, [dstq+strideq*0]
    lea                  r5, [dstq+strideq*2]
    movq                 m2, [r5  +strideq*1]
    movhps               m2, [r5  +strideq*0]
    mova                 m5, [o(pixel_10bpc_max)]
    pmulhrsw             m0, m4
    pmulhrsw             m1, m4
    pxor                 m4, m4
    mova          [cq+16*0], m4
    mova          [cq+16*1], m4
    mova          [cq+16*2], m4
    mova          [cq+16*3], m4
    paddw                m0, m2
    paddw                m1, m3
    pmaxsw               m0, m4
    pmaxsw               m1, m4
    pminsw               m0, m5
    pminsw               m1, m5
    movhps [dstq+strideq*0], m1
    movq   [dstq+strideq*1], m1
    movhps [r5  +strideq*0], m0
    movq   [r5  +strideq*1], m0
    RET

INV_TXFM_4X4_FN identity, dct
INV_TXFM_4X4_FN identity, adst
INV_TXFM_4X4_FN identity, flipadst
INV_TXFM_4X4_FN identity, identity

cglobal iidentity_4x4_internal_16bpc, 0, 7, 8, dst, stride, c, eob, tx2
    mova                 m3, [o(pd_5793)]
    pmulld               m0, m3, [cq+16*0]
    pmulld               m1, m3, [cq+16*1]
    pmulld               m2, m3, [cq+16*2]
    pmulld               m3,     [cq+16*3]
    mova                 m5, [o(pd_2048)]
    REPX      {paddd x, m5}, m0, m1, m2, m3
    REPX      {psrad x, 12}, m0, m1, m2, m3
    packssdw             m0, m1
    packssdw             m2, m3
    ; transpose
    punpckhwd            m3, m0, m2
    punpcklwd            m0, m2
    punpckhwd            m1, m0, m3
    punpcklwd            m0, m3
    ; m0 = out0 out1
    ; m1 = out2 out3
    ; m5 = pd_2048
    jmp                tx2q
.pass2:
    ; m0 = in0 in1
    ; m1 = in2 in3
    ; m5 = pd_2048
    mova                 m4, [o(pw_1697x8)]
    movq                 m2, [dstq+strideq*0]
    movhps               m2, [dstq+strideq*1]
    lea                  r5, [dstq+strideq*2]
    pmulhrsw             m3, m4, m0
    pmulhrsw             m4, m1
    paddsw               m0, m3
    paddsw               m1, m4
    movq                 m3, [r5  +strideq*0]
    movhps               m3, [r5  +strideq*1]
    mova                 m4, [o(pixel_10bpc_max)]
    packssdw             m5, m5 ; pw_2048
    pmulhrsw             m0, m5
    pmulhrsw             m1, m5
    pxor                 m5, m5
    mova          [cq+16*0], m5
    mova          [cq+16*1], m5
    mova          [cq+16*2], m5
    mova          [cq+16*3], m5
    paddw                m0, m2
    paddw                m1, m3
    pmaxsw               m0, m5
    pmaxsw               m1, m5
    pminsw               m0, m4
    pminsw               m1, m4
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movq   [r5  +strideq*0], m1
    movhps [r5  +strideq*1], m1
    RET
