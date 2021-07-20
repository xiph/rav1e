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
pw_4x2048_4xm2048: times 4 dw 2048
                   times 4 dw -2048
pw_4xm2048_4x2048: times 4 dw -2048
                   times 4 dw 2048
pw_2048:         times 8 dw  2048
pd_3803:         times 4 dd  3803
pw_4096:         times 8 dw  4096
pd_5793:         times 4 dd  5793
pd_6144:         times 4 dd  6144
pw_1697x8:       times 8 dw  1697*8
pw_2896x8:       times 8 dw  2896*8
pw_1697x16:      times 8 dw  1697*16
pw_16384:        times 8 dw 16384
pixel_10bpc_max: times 8 dw  0x03ff

pw_1567_3784:    times 4 dw  1567,  3784
pw_m3784_1567:   times 4 dw -3784,  1567

clip_min: times 4 dd -0x20000
clip_max: times 4 dd  0x1ffff

cextern inv_txfm_add_dct_dct_4x4_8bpc_ssse3
cextern iadst_4x4_internal_8bpc_ssse3.main
cextern idct_4x8_internal_8bpc_ssse3.main
cextern iadst_4x8_internal_8bpc_ssse3.main
cextern idct_16x4_internal_8bpc_ssse3.main
cextern iadst_16x4_internal_8bpc_ssse3.main
cextern iadst_16x4_internal_8bpc_ssse3.main_pass2_end
cextern idct_8x4_internal_8bpc_ssse3.main
cextern iadst_8x4_internal_8bpc_ssse3.main

tbl_4x16_2d: db 0, 13, 29, 45
tbl_4x16_h: db 0, 16, 32, 48
tbl_4x16_v: db 0, 4, 8, 12

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

%macro INV_TXFM_FN 4-5+ 8 ; type1, type2, eob_offset, size, mmsize/stack
cglobal inv_txfm_add_%1_%2_%4_16bpc, 4, 7, %5, dst, stride, c, eob, tx2
    %define %%p1 m(i%1_%4_internal_16bpc)
%if ARCH_X86_32
    LEA                  r6, $$
%endif
%if has_epilogue
%ifidn %1_%2, dct_dct
    test               eobd, eobd
    jz %%end
%endif
    lea                tx2q, [o(m(i%2_%4_internal_16bpc).pass2)]
%ifnum %3
%if %3
    add                eobd, %3
%endif
%else
    lea                  r5, [o(%3)]
%endif
    call %%p1
    RET
%%end:
%else
    ; Jump to the 1st txfm function if we're not taking the fast path, which
    ; in turn performs an indirect jump to the 2nd txfm function.
    lea                tx2q, [o(m(i%2_%4_internal_16bpc).pass2)]
%ifnum %3
%if %3
    add                eobd, %3
%endif
%else
    lea                  r5, [o(%3)]
%endif
%ifidn %1_%2, dct_dct
    test               eobd, eobd
    jnz %%p1
%else
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
    TAIL_CALL m(iadst_4x4_internal_16bpc).end
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

cglobal idct_4x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    mova                 m0, [cq+16*0]
    mova                 m1, [cq+16*1]
    mova                 m2, [cq+16*2]
    mova                 m3, [cq+16*3]
    mova                 m5, [o(pd_2048)]
    call .pass1_main
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
.pass1_main:
    IDCT4_1D              0, 1, 2, 3, 4, 6, 7, 5
    ret
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

cglobal iadst_4x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call .main
    packssdw             m0, m2            ; out0 out1
    packssdw             m1, m4            ; out2 out3
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
    mova                 m5, [cq+16*0]
    lea                  r3, [cq+16*1]
.main2:
    mova                 m0, [o(pd_1321)]  ; SINPI_1_9
    mova                 m2, [o(pd_2482)]  ; SINPI_2_9
    mova                 m6, [o(pd_3803)]  ; SINPI_4_9
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
    pmulld               m3, [r3]          ; -s[3] = -SINPI_3_9 * T[1]
    mova                 m5, [o(pd_2048)]
    REPX      {paddd x, m5}, m0, m1        ; {s[0], s[2]} + 2048
    paddd                m4, m0, m2        ; x[3]  = s[0] + s[1]
    psubd                m2, m3            ; x[1]  = s[1] + s[3]
    psubd                m0, m3            ; x[0]  = s[0] + s[3]
    paddd                m4, m3            ; x[3] -= s[3]
    paddd                m2, m5            ; x[1] + 2048
    REPX      {psrad x, 12}, m0, m2, m1, m4
    ret


INV_TXFM_4X4_FN flipadst, dct
INV_TXFM_4X4_FN flipadst, adst
INV_TXFM_4X4_FN flipadst, flipadst
INV_TXFM_4X4_FN flipadst, identity

cglobal iflipadst_4x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call m(iadst_4x4_internal_16bpc).main
    packssdw             m0, m2            ; out0 out1
    packssdw             m1, m4            ; out2 out3
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

cglobal iidentity_4x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
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

%macro INV_TXFM_4X8_FN 2-3 0 ; type1, type2
    INV_TXFM_FN          %1, %2, %3, 4x8
%ifidn %1_%2, dct_dct
    imul                r5d, [cq], 2896
    mov                [cq], eobd ; 0
    mov                 r3d, 2
    add                 r5d, 2048
    sar                 r5d, 12
    imul                r5d, 2896
    add                 r5d, 2048
    sar                 r5d, 12
.end:
    imul                r5d, 2896
    add                 r5d, 34816
    movd                 m0, r5d
    pshuflw              m0, m0, q1111
    punpcklqdq           m0, m0
    pxor                 m4, m4
    mova                 m3, [o(pixel_10bpc_max)]
    lea                  r2, [strideq*3]
.loop:
    movq                 m1, [dstq+strideq*0]
    movq                 m2, [dstq+strideq*2]
    movhps               m1, [dstq+strideq*1]
    movhps               m2, [dstq+r2]
    paddw                m1, m0
    paddw                m2, m0
    REPX     {pminsw x, m3}, m1, m2
    REPX     {pmaxsw x, m4}, m1, m2
    movq   [dstq+strideq*0], m1
    movhps [dstq+strideq*1], m1
    movq   [dstq+strideq*2], m2
    movhps [dstq+r2       ], m2
    lea                dstq, [dstq+strideq*4]
    dec                 r3d
    jg .loop
    RET
%endif
%endmacro

INV_TXFM_4X8_FN dct, dct
INV_TXFM_4X8_FN dct, identity, 9
INV_TXFM_4X8_FN dct, adst
INV_TXFM_4X8_FN dct, flipadst

cglobal idct_4x8_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
    mova                 m5, [o(pd_2048)]
%if ARCH_X86_64
    xor                 r5d, r5d
    cmp                eobd, 13
    setge               r5b
%else
    mov                 r5d, 1
    cmp                eobd, 13
    sbb                 r5d, 0
%endif
    shl                 r5d, 4
.loop_pass1:
    mova                 m3, [o(pd_2896)]
    pmulld               m0, m3, [cq+32*0+r5]
    pmulld               m1, m3, [cq+32*1+r5]
    pmulld               m2, m3, [cq+32*2+r5]
    pmulld               m3, [cq+32*3+r5]
    REPX      {paddd x, m5}, m0, m1, m2, m3
    REPX      {psrad x, 12}, m0, m1, m2, m3
    call m(idct_4x4_internal_16bpc).pass1_main
    packssdw             m0, m1     ; out0 out1
    packssdw             m4, m2     ; out2 out3
    test                r5d, r5d
    jz .end_pass1
    mova       [cq+32*0+16], m0
    mova       [cq+32*1+16], m4
    xor                 r5d, r5d
    jmp .loop_pass1
.end_pass1:
    punpckhwd            m2, m0, m4
    punpcklwd            m0, m4
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    mova                 m2, [cq+32*0+16]
    mova                 m6, [cq+32*1+16]
    punpckhwd            m4, m2, m6
    punpcklwd            m2, m6
    punpckhwd            m3, m2, m4
    punpcklwd            m2, m4
    ; m0-3 = packed & transposed output
    jmp                tx2q
.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(idct_4x8_internal_8bpc, _ssse3).main
    ; m0-3 is now out0/1,3/2,4/5,7/6
    mova                 m4, [o(pw_2048)]
    shufps               m1, m1, q1032
    shufps               m3, m3, q1032
.end:
    REPX   {pmulhrsw x, m4}, m0, m1, m2, m3
    pxor                 m4, m4
    REPX {mova [cq+16*x], m4}, 0, 1, 2, 3, 4, 5, 6, 7
    mova                 m7, [o(pixel_10bpc_max)]
    lea                  r2, [strideq*3]
    movq                 m5, [dstq+strideq*0]
    movq                 m6, [dstq+strideq*2]
    movhps               m5, [dstq+strideq*1]
    movhps               m6, [dstq+r2]
    lea                  r4, [dstq+strideq*4]
    paddw                m0, m5
    paddw                m1, m6
    movq                 m5, [r4+strideq*0]
    movq                 m6, [r4+strideq*2]
    movhps               m5, [r4+strideq*1]
    movhps               m6, [r4+r2]
    paddw                m2, m5
    paddw                m3, m6
    REPX     {pminsw x, m7}, m0, m1, m2, m3
    REPX     {pmaxsw x, m4}, m0, m1, m2, m3
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movq   [dstq+strideq*2], m1
    movhps [dstq+r2       ], m1
    movq   [r4  +strideq*0], m2
    movhps [r4  +strideq*1], m2
    movq   [r4  +strideq*2], m3
    movhps [r4  +r2       ], m3
    RET

INV_TXFM_4X8_FN adst, dct
INV_TXFM_4X8_FN adst, adst
INV_TXFM_4X8_FN adst, flipadst
INV_TXFM_4X8_FN adst, identity, 9

cglobal iadst_4x8_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call .pass1_main
    punpckhwd            m2, m0, m1
    punpcklwd            m0, m1
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    mova                 m2, [cq+32*2+16]
    mova                 m6, [cq+32*3+16]
    punpckhwd            m4, m2, m6
    punpcklwd            m2, m6
    punpckhwd            m3, m2, m4
    punpcklwd            m2, m4
    ; m0-3 = packed & transposed output
    jmp                tx2q
.pass1_main:
%undef cmp
%if ARCH_X86_64
    xor                 r5d, r5d
    cmp                eobd, 13
    setge               r5b
%else
    mov                 r5d, 1
    cmp                eobd, 13
    sbb                 r5d, 0
%endif
    shl                 r5d, 4
    lea                  r3, [cq+32*1+16]
.loop_pass1:
    mova                 m0, [o(pd_2048)]
    mova                 m3, [o(pd_2896)]
    pmulld               m5, m3, [cq+32*0+r5]
    pmulld               m2, m3, [cq+32*1+r5]
    pmulld               m1, m3, [cq+32*2+r5]
    pmulld               m3, [cq+32*3+r5]
    REPX      {paddd x, m0}, m5, m2, m1, m3
    REPX      {psrad x, 12}, m5, m2, m1, m3
    mova               [r3], m2
    call m(iadst_4x4_internal_16bpc).main2
    packssdw             m0, m2            ; out0 out1
    packssdw             m1, m4            ; out2 out3
    test                r5d, r5d
    jz .end_pass1
    mova       [cq+32*2+16], m0
    mova       [cq+32*3+16], m1
    xor                 r5d, r5d
    jmp .loop_pass1
.end_pass1:
    ret
.pass2:
    shufps               m0, m0, q1032
    shufps               m1, m1, q1032
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_4x8_internal_8bpc, _ssse3).main
    mova                 m4, [o(pw_4x2048_4xm2048)]
    jmp m(idct_4x8_internal_16bpc).end

INV_TXFM_4X8_FN flipadst, dct
INV_TXFM_4X8_FN flipadst, adst
INV_TXFM_4X8_FN flipadst, flipadst
INV_TXFM_4X8_FN flipadst, identity, 9

cglobal iflipadst_4x8_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call m(iadst_4x8_internal_16bpc).pass1_main
    punpcklwd            m2, m1, m0
    punpckhwd            m1, m0
    punpcklwd            m0, m1, m2
    punpckhwd            m1, m2
    mova                 m6, [cq+32*2+16]
    mova                 m2, [cq+32*3+16]
    punpcklwd            m4, m2, m6
    punpckhwd            m2, m6
    punpckhwd            m3, m2, m4
    punpcklwd            m2, m4
    ; m0-3 = packed & transposed output
    jmp                tx2q
.pass2:
    shufps               m0, m0, q1032
    shufps               m1, m1, q1032
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_4x8_internal_8bpc, _ssse3).main
    mova                 m4, m0
    mova                 m5, m1
    pshufd               m0, m3, q1032
    pshufd               m1, m2, q1032
    pshufd               m2, m5, q1032
    pshufd               m3, m4, q1032
    mova                 m4, [o(pw_4xm2048_4x2048)]
    jmp m(idct_4x8_internal_16bpc).end

INV_TXFM_4X8_FN identity, dct
INV_TXFM_4X8_FN identity, adst
INV_TXFM_4X8_FN identity, flipadst
INV_TXFM_4X8_FN identity, identity, 3

cglobal iidentity_4x8_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
    mova                 m5, [o(pd_2048)]
    mova                 m4, [o(pd_2896)]
    mova                 m6, [o(pd_5793)]
    ; clear m7 in case we skip the bottom square
    pxor                 m7, m7
%if ARCH_X86_64
    xor                 r5d, r5d
    cmp                eobd, 16
    setge               r5b
%else
    mov                 r5d, 1
    cmp                eobd, 16
    sbb                 r5d, 0
%endif
    shl                 r5d, 4
.loop_pass1:
    pmulld               m0, m4, [cq+32*0+r5]
    pmulld               m1, m4, [cq+32*1+r5]
    pmulld               m2, m4, [cq+32*2+r5]
    pmulld               m3, m4, [cq+32*3+r5]
    REPX      {paddd x, m5}, m0, m1, m2, m3
    REPX      {psrad x, 12}, m0, m1, m2, m3
    REPX     {pmulld x, m6}, m0, m1, m2, m3
    REPX      {paddd x, m5}, m0, m1, m2, m3
    REPX      {psrad x, 12}, m0, m1, m2, m3
    packssdw             m0, m1
    packssdw             m2, m3
    test                r5d, r5d
    jz .end_pass1
    mova       [cq+32*0+16], m0
    mova                 m7, m2
    xor                 r5d, r5d
    jmp .loop_pass1
.end_pass1:
    punpckhwd            m4, m0, m2
    punpcklwd            m0, m2
    punpckhwd            m1, m0, m4
    punpcklwd            m0, m4
    mova                 m2, [cq+32*0+16]
    punpckhwd            m4, m2, m7
    punpcklwd            m2, m7
    punpckhwd            m3, m2, m4
    punpcklwd            m2, m4
    ; m0-3 = packed & transposed output
    jmp                tx2q
.pass2:
    mova                 m4, [o(pw_4096)]
    jmp m(idct_4x8_internal_16bpc).end

%macro INV_TXFM_4X16_FN 2-3 2d ; type1, type2
    INV_TXFM_FN          %1, %2, tbl_4x16_%3, 4x16
%ifidn %1_%2, dct_dct
    imul                r5d, [cq], 2896
    mov                [cq], eobd ; 0
    mov                 r3d, 4
    add                 r5d, 6144
    sar                 r5d, 13
    jmp m(inv_txfm_add_dct_dct_4x8_16bpc).end
%endif
%endmacro

INV_TXFM_4X16_FN dct, dct
INV_TXFM_4X16_FN dct, identity, v
INV_TXFM_4X16_FN dct, adst
INV_TXFM_4X16_FN dct, flipadst

cglobal idct_4x16_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
%if ARCH_X86_32
    mov                 r5m, r6d
%endif
    mov                 r6d, 4
.zero_loop:
    dec                 r6d
    cmp                eobb, byte [r5+r6]
    jl .zero_loop
    mov                 r5d, r6d
    shl                 r5d, 4
%if ARCH_X86_32
    ; restore pic-ptr
    mov                  r6, r5m
%endif
    mova                 m5, [o(pd_2048)]
.loop_pass1:
    mova                 m0, [cq+64*0+r5]
    mova                 m1, [cq+64*1+r5]
    mova                 m2, [cq+64*2+r5]
    mova                 m3, [cq+64*3+r5]
    call m(idct_4x4_internal_16bpc).pass1_main
    pcmpeqd              m3, m3
    REPX      {psubd x, m3}, m0, m1, m4, m2
    REPX       {psrad x, 1}, m0, m1, m4, m2
    packssdw             m0, m1     ; out0 out1
    packssdw             m4, m2     ; out2 out3
    punpckhwd            m2, m0, m4
    punpcklwd            m0, m4
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    test                r5d, r5d
    jz .end_pass1
    mova       [cq+64*0+r5], m0
    mova       [cq+64*1+r5], m1
    sub                 r5d, 16
    jmp .loop_pass1
.end_pass1:
    mova                 m2, [cq+64*0+16]
    mova                 m3, [cq+64*1+16]
    mova                 m4, [cq+64*0+32]
    mova                 m5, [cq+64*1+32]
    mova                 m6, [cq+64*0+48]
    mova                 m7, [cq+64*1+48]
    ; m0-7 = packed & transposed output
    jmp                tx2q
.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(idct_16x4_internal_8bpc, _ssse3).main
    ; m0-6 is out0-13 [with odd registers having inversed output]
    ; [coeffq+16*7] has out15/14
    mova                 m7, [o(pw_2048)]
    REPX   {pmulhrsw x, m7}, m0, m1, m2, m3, m4, m5, m6
    pmulhrsw             m7, [cq+16*7]
    REPX {shufps x, x, q1032}, m1, m3, m5, m7
    mova          [cq+16*0], m4
    mova          [cq+16*1], m5
    mova          [cq+16*2], m6
    mova          [cq+16*3], m7
.end:
    pxor                 m4, m4
    REPX {mova [cq+16*x], m4}, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    mova                 m7, [o(pixel_10bpc_max)]
    mov                 r5d, 2
    lea                  r3, [strideq*3]
.loop:
    movq                 m5, [dstq+strideq*0]
    movq                 m6, [dstq+strideq*2]
    movhps               m5, [dstq+strideq*1]
    movhps               m6, [dstq+r3]
    lea                  r4, [dstq+strideq*4]
    paddw                m0, m5
    paddw                m1, m6
    movq                 m5, [r4+strideq*0]
    movq                 m6, [r4+strideq*2]
    movhps               m5, [r4+strideq*1]
    movhps               m6, [r4+r3]
    paddw                m2, m5
    paddw                m3, m6
    REPX     {pminsw x, m7}, m0, m1, m2, m3
    REPX     {pmaxsw x, m4}, m0, m1, m2, m3
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    movq   [dstq+strideq*2], m1
    movhps [dstq+r3       ], m1
    movq   [r4  +strideq*0], m2
    movhps [r4  +strideq*1], m2
    movq   [r4  +strideq*2], m3
    movhps [r4  +r3       ], m3
    dec                 r5d
    jz .end2
    lea                dstq, [dstq+strideq*8]
    mova                 m0, [cq+0*16]
    mova                 m1, [cq+1*16]
    mova                 m2, [cq+2*16]
    mova                 m3, [cq+3*16]
    REPX {mova [cq+x*16], m4}, 0, 1, 2, 3
    jmp .loop
.end2:
    RET

INV_TXFM_4X16_FN adst, dct
INV_TXFM_4X16_FN adst, adst
INV_TXFM_4X16_FN adst, flipadst
INV_TXFM_4X16_FN adst, identity, v

cglobal iadst_4x16_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
%if ARCH_X86_32
    mov                 r5m, r6d
%endif
    mov                 r6d, 4
.zero_loop:
    dec                 r6d
    cmp                eobb, byte [r6+r5]
    jl .zero_loop
    mov                 r5d, r6d
    shl                 r5d, 4
%if ARCH_X86_32
    ; restore pic-ptr
    mov                  r6, r5m
%endif
.loop_pass1:
    mova                 m5, [cq+64*0+r5]
    lea                  r3, [cq+64*1+r5]
    mova                 m1, [cq+64*2+r5]
    mova                 m3, [cq+64*3+r5]
    call m(iadst_4x4_internal_16bpc).main2
    pcmpeqd              m3, m3
    REPX      {psubd x, m3}, m0, m2, m1, m4
    REPX       {psrad x, 1}, m0, m2, m1, m4
    packssdw             m0, m2            ; out0 out1
    packssdw             m1, m4            ; out2 out3
    punpckhwd            m2, m0, m1
    punpcklwd            m0, m1
    punpckhwd            m1, m0, m2
    punpcklwd            m0, m2
    test                r5d, r5d
    jz m(idct_4x16_internal_16bpc).end_pass1
    mova       [cq+64*0+r5], m0
    mova       [cq+64*1+r5], m1
    sub                 r5d, 16
    jmp .loop_pass1
.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_16x4_internal_8bpc, _ssse3).main
    call m_suffix(iadst_16x4_internal_8bpc, _ssse3).main_pass2_end
    ; m7/5/2/4 = out4/-11,-5/10,6/-9,-7/8
    ; m0/3 & cq6/7 = out0/-15,-3/12,-1/14,2/-13
    mova                 m1, [o(pw_4x2048_4xm2048)]
    REPX   {pmulhrsw x, m1}, m7, m2, m0
    pshufd               m6, m1, q1032  ; 4x-2048,4x2048
    pmulhrsw             m1, [cq+16*7]
    REPX   {pmulhrsw x, m6}, m5, m4, m3
    pmulhrsw             m6, [cq+16*6]
    ; m7/5/2/4 = out4/11,5/10,6/9,7/8
    ; m0/3/6/1 = out0/15,3/12,1/14,2/13
    ; output should be as 0-3 for out0-7, and cq+0-3*16 for out8-15
    movhps         [cq+0*8], m4
    movhps         [cq+1*8], m2
    movhps         [cq+2*8], m5
    movhps         [cq+3*8], m7
    movhps         [cq+4*8], m3
    movhps         [cq+5*8], m1
    movhps         [cq+6*8], m6
    movhps         [cq+7*8], m0
    punpcklqdq           m0, m6
    punpcklqdq           m1, m3
    punpcklqdq           m3, m2, m4
    punpcklqdq           m2, m7, m5
    jmp m(idct_4x16_internal_16bpc).end

INV_TXFM_4X16_FN flipadst, dct
INV_TXFM_4X16_FN flipadst, adst
INV_TXFM_4X16_FN flipadst, flipadst
INV_TXFM_4X16_FN flipadst, identity, v

cglobal iflipadst_4x16_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
%if ARCH_X86_32
    mov                 r5m, r6d
%endif
    mov                 r6d, 4
.zero_loop:
    dec                 r6d
    cmp                eobb, byte [r5+r6]
    jl .zero_loop
    mov                 r5d, r6d
    shl                 r5d, 4
%if ARCH_X86_32
    ; restore pic-ptr
    mov                  r6, r5m
%endif
.loop_pass1:
    mova                 m5, [cq+64*0+r5]
    lea                  r3, [cq+64*1+r5]
    mova                 m1, [cq+64*2+r5]
    mova                 m3, [cq+64*3+r5]
    call m(iadst_4x4_internal_16bpc).main2
    pcmpeqd              m3, m3
    REPX      {psubd x, m3}, m0, m2, m1, m4
    REPX       {psrad x, 1}, m0, m2, m1, m4
    packssdw             m0, m2            ; out3 out2
    packssdw             m1, m4            ; out1 out0
    punpcklwd            m2, m1, m0
    punpckhwd            m1, m0
    punpcklwd            m0, m1, m2
    punpckhwd            m1, m2
    test                r5d, r5d
    jz m(idct_4x16_internal_16bpc).end_pass1
    mova       [cq+64*0+r5], m0
    mova       [cq+64*1+r5], m1
    sub                 r5d, 16
    jmp .loop_pass1
.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_16x4_internal_8bpc, _ssse3).main
    call m_suffix(iadst_16x4_internal_8bpc, _ssse3).main_pass2_end
    ; m7/5/2/4 = out11/-4,-10/5,9/-6,-8/7
    ; m0/3 & cq6/7 = out15/-0,-12/3,-14/1,13/-2
    mova                 m1, [o(pw_4x2048_4xm2048)]
    REPX   {pmulhrsw x, m1}, m7, m2, m0
    pshufd               m6, m1, q1032  ; 4x-2048,4x2048
    pmulhrsw             m1, [cq+16*7]
    REPX   {pmulhrsw x, m6}, m5, m4, m3
    pmulhrsw             m6, [cq+16*6]
    ; m7/5/2/4 = out11/4,10/5,9/6,8/7
    ; m0/3/6/1 = out15/0,12/3,14/1,13/2
    ; output should be as 0-3 for out0-7, and cq+0-3*16 for out8-15
    movq           [cq+0*8], m4
    movq           [cq+1*8], m2
    movq           [cq+2*8], m5
    movq           [cq+3*8], m7
    movq           [cq+4*8], m3
    movq           [cq+5*8], m1
    movq           [cq+6*8], m6
    movq           [cq+7*8], m0
    punpckhqdq           m0, m6
    punpckhqdq           m1, m3
    punpckhqdq           m3, m2, m4
    punpckhqdq           m2, m7, m5
    jmp m(idct_4x16_internal_16bpc).end

INV_TXFM_4X16_FN identity, dct, h
INV_TXFM_4X16_FN identity, adst, h
INV_TXFM_4X16_FN identity, flipadst, h
INV_TXFM_4X16_FN identity, identity

cglobal iidentity_4x16_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
%undef cmp
%if ARCH_X86_32
    mov                 r5m, r6d
%endif
    mov                 r6d, 4
.zero_loop:
    dec                 r6d
    cmp                eobb, byte [r5+r6]
    jl .zero_loop
    mov                 r5d, r6d
    shl                 r5d, 4
%if ARCH_X86_32
    ; restore pic-ptr
    mov                  r6, r5m
%endif
    mova                 m5, [o(pd_6144)]
    mova                 m4, [o(pd_5793)]
.loop_pass1:
    pmulld               m0, m4, [cq+64*0+r5]
    pmulld               m1, m4, [cq+64*1+r5]
    pmulld               m2, m4, [cq+64*2+r5]
    pmulld               m3, m4, [cq+64*3+r5]
    REPX      {paddd x, m5}, m0, m1, m2, m3
    REPX      {psrad x, 13}, m0, m1, m2, m3
    packssdw             m0, m1
    packssdw             m2, m3
    punpckhwd            m3, m0, m2
    punpcklwd            m0, m2
    punpckhwd            m1, m0, m3
    punpcklwd            m0, m3
    test                r5d, r5d
    jz m(idct_4x16_internal_16bpc).end_pass1
    mova       [cq+64*0+r5], m0
    mova       [cq+64*1+r5], m1
    sub                 r5d, 16
    jmp .loop_pass1
.pass2:
    mova          [cq+16*4], m0
    mova          [cq+16*5], m1
    mova          [cq+16*6], m2
    mova          [cq+16*7], m7
    mova                 m0, [o(pw_1697x16)]
    mova                 m7, [o(pw_2048)]
    pmulhrsw             m1, m0, m4
    pmulhrsw             m2, m0, m5
    REPX      {paddsw x, x}, m4, m5
    paddsw               m4, m1
    paddsw               m5, m2
    REPX   {pmulhrsw x, m7}, m4, m5
    mova          [cq+16*0], m4
    mova          [cq+16*1], m5
    mova                 m4, [cq+16*7]
    pmulhrsw             m1, m0, m6
    pmulhrsw             m2, m0, m4
    REPX      {paddsw x, x}, m6, m4
    paddsw               m6, m1
    paddsw               m4, m2
    REPX   {pmulhrsw x, m7}, m6, m4
    mova          [cq+16*2], m6
    mova          [cq+16*3], m4
    mova                 m4, [cq+16*4]
    mova                 m1, [cq+16*5]
    mova                 m2, [cq+16*6]
    pmulhrsw             m5, m0, m2
    pmulhrsw             m6, m0, m3
    REPX      {paddsw x, x}, m2, m3
    paddsw               m2, m5
    paddsw               m3, m6
    pmulhrsw             m6, m0, m1
    pmulhrsw             m0, m4
    REPX      {paddsw x, x}, m1, m4
    paddsw               m1, m6
    paddsw               m0, m4
    REPX   {pmulhrsw x, m7}, m2, m3, m1, m0
    jmp m(idct_4x16_internal_16bpc).end

%macro INV_TXFM_8X4_FN 2 ; type1, type2
%if ARCH_X86_64
    INV_TXFM_FN          %1, %2, 0, 8x4, 14
%else
    INV_TXFM_FN          %1, %2, 0, 8x4, 8, 0-4*16
%endif
%ifidn %1_%2, dct_dct
    imul                r5d, [cq], 2896
    mov                [cq], eobd ; 0
    add                 r5d, 2048
    sar                 r5d, 12
    imul                r5d, 2896
    add                 r5d, 2048
    sar                 r5d, 12
    imul                r5d, 2896
    add                 r5d, 34816
    movd                 m0, r5d
    pshuflw              m0, m0, q1111
    punpcklqdq           m0, m0
    mova                 m6, [o(pixel_10bpc_max)]
    pxor                 m5, m5
    lea                  r2, [strideq*3]
    mova                 m1, [dstq+strideq*0]
    mova                 m2, [dstq+strideq*1]
    mova                 m3, [dstq+strideq*2]
    mova                 m4, [dstq+r2]
    REPX      {paddw x, m0}, m1, m2, m3, m4
    REPX     {pmaxsw x, m5}, m1, m2, m3, m4
    REPX     {pminsw x, m6}, m1, m2, m3, m4
    mova   [dstq+strideq*0], m1
    mova   [dstq+strideq*1], m2
    mova   [dstq+strideq*2], m3
    mova   [dstq+r2       ], m4
    RET
%endif
%endmacro

INV_TXFM_8X4_FN dct, dct
INV_TXFM_8X4_FN dct, identity
INV_TXFM_8X4_FN dct, adst
INV_TXFM_8X4_FN dct, flipadst

cglobal idct_8x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call .load
    call .main_pass1
.pack_transpose:
    packssdw             m0, m1
    packssdw             m2, m3
    packssdw             m4, m5
    packssdw             m6, m7
.transpose:
    ; transpose
    punpckhwd            m5, m0, m4
    punpcklwd            m0, m4
    punpckhwd            m4, m2, m6
    punpcklwd            m2, m6

    punpckhwd            m3, m0, m2
    punpcklwd            m0, m2
    punpckhwd            m7, m5, m4
    punpcklwd            m5, m4

    punpckhwd            m1, m0, m5
    punpcklwd            m0, m5
    punpcklwd            m2, m3, m7
    punpckhwd            m3, m7
    ; m0-3 = packed & transposed output
    jmp                tx2q
.load:
    mova                 m7, [o(pd_2896)]
    pmulld               m0, m7, [cq+0*16]
    pmulld               m1, m7, [cq+1*16]
    pmulld               m2, m7, [cq+2*16]
    pmulld               m3, m7, [cq+3*16]
    pmulld               m4, m7, [cq+4*16]
    pmulld               m5, m7, [cq+5*16]
    pmulld               m6, m7, [cq+6*16]
    pmulld               m7, [cq+7*16]
%if ARCH_X86_64
    mova                 m8, [o(pd_2048)]
    REPX      {paddd x, m8}, m0, m1, m2, m3, m4, m5, m6, m7
%else
    mova          [cq+0*16], m7
    mova                 m7, [o(pd_2048)]
    REPX      {paddd x, m7}, m0, m1, m2, m3, m4, m5, m6
    paddd                m7, [cq+0*16]
%endif
    REPX      {psrad x, 12}, m0, m1, m2, m3, m4, m5, m6, m7
    ret
.main_pass1:
%if ARCH_X86_64
    mova                m11, [o(pd_2048)]
    mova                m12, [o(clip_min)]
    mova                m13, [o(clip_max)]
    ITX_MULSUB_2D         5, 3, 8, 9, 10, 11, 3406, 2276 ; t5a t6a
    ITX_MULSUB_2D         1, 7, 8, 9, 10, 11,  799, 4017 ; t4a t7a
    ITX_MULSUB_2D         2, 6, 8, 9, 10, 11, 1567, 3784 ; t2  t3
    paddd                m8, m1, m5 ; t4
    psubd                m1, m5     ; t5a
    paddd                m9, m7, m3 ; t7
    psubd                m7, m3     ; t6a
    mova                 m3, [o(pd_2896)]
    REPX    {pmaxsd x, m12}, m1, m8, m7, m9
    REPX    {pminsd x, m13}, m1, m8, m7, m9
    REPX    {pmulld x, m3 }, m0, m4, m7, m1
    paddd                m0, m11
    paddd                m7, m11
    psubd                m5, m0, m4
    paddd                m0, m4
    psubd                m4, m7, m1
    paddd                m7, m1
    REPX    {psrad  x, 12 }, m5, m0, m4, m7
    psubd                m3, m0, m6 ; dct4 out3
    paddd                m0, m6     ; dct4 out0
    paddd                m6, m5, m2 ; dct4 out1
    psubd                m5, m2     ; dct4 out2
    REPX    {pmaxsd x, m12}, m0, m6, m5, m3
    REPX    {pminsd x, m13}, m0, m6, m5, m3

    paddd                m1, m6, m7 ; out1
    psubd                m6, m7     ; out6
    psubd                m7, m0, m9 ; out7
    paddd                m0, m9     ; out0
    paddd                m2, m5, m4 ; out2
    psubd                m5, m4     ; out5
    psubd                m4, m3, m8 ; out4
    paddd                m3, m8     ; out3
%else
    mova [rsp+0*16+2*gprsize], m0
    mova [rsp+1*16+2*gprsize], m2
    mova [rsp+2*16+2*gprsize], m4
    mova [rsp+3*16+2*gprsize], m6
    mova                 m0, [o(pd_2048)]
    ITX_MULSUB_2D         5, 3, 2, 4, 6, 0, 3406, 2276 ; t5a t6a
    ITX_MULSUB_2D         1, 7, 2, 4, 6, 0,  799, 4017 ; t4a t7a
    paddd                m2, m1, m5 ; t4
    psubd                m1, m5     ; t5a
    paddd                m4, m7, m3 ; t7
    psubd                m7, m3     ; t6a
    mova                 m6, [o(clip_min)]
    REPX    {pmaxsd x, m6 }, m1, m2, m7, m4
    mova                 m6, [o(clip_max)]
    REPX    {pminsd x, m6 }, m1, m2, m7, m4
    mova                 m6, [rsp+3*16+2*gprsize]
    mova [rsp+3*16+2*gprsize], m2
    mova                 m2, [rsp+1*16+2*gprsize]
    mova [rsp+1*16+2*gprsize], m4

    ITX_MULSUB_2D         2, 6, 4, 3, 5, 0, 1567, 3784 ; t2  t3
    mova                 m3, [o(pd_2896)]
    mova                 m5, [rsp+0*16+2*gprsize]
    mova                 m4, [rsp+2*16+2*gprsize]
    REPX    {pmulld x, m3 }, m5, m4, m7, m1
    paddd                m7, m0
    paddd                m0, m5

    psubd                m5, m0, m4
    paddd                m0, m4
    psubd                m4, m7, m1
    paddd                m7, m1
    REPX    {psrad  x, 12 }, m5, m0, m4, m7
    psubd                m3, m0, m6 ; dct4 out3
    paddd                m0, m6     ; dct4 out0
    paddd                m6, m5, m2 ; dct4 out1
    psubd                m5, m2     ; dct4 out2

    mova                 m1, [o(clip_min)]
    REPX    {pmaxsd x, m1 }, m0, m6, m5, m3
    mova                 m1, [o(clip_max)]
    REPX    {pminsd x, m1 }, m0, m6, m5, m3

    paddd                m1, m6, m7 ; out1
    psubd                m6, m7     ; out6
    mova [rsp+0*16+2*gprsize], m6
    mova                 m6, [rsp+1*16+2*gprsize]
    psubd                m7, m0, m6 ; out7
    paddd                m0, m6     ; out0
    paddd                m2, m5, m4 ; out2
    psubd                m5, m4     ; out5
    mova                 m6, [rsp+3*16+2*gprsize]
    psubd                m4, m3, m6 ; out4
    paddd                m3, m6     ; out3
    mova                 m6, [rsp+0*16+2*gprsize]
%endif
    ret

.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(idct_8x4_internal_8bpc, _ssse3).main
.end:
    lea                  r3, [strideq*3]
.end2:
    ; output is in m0-3
    mova                 m4, [o(pw_2048)]
.end3:
    REPX   {pmulhrsw x, m4}, m0, m1, m2, m3
    pxor                 m4, m4
    REPX {mova [cq+16*x], m4}, 0, 1, 2, 3, 4, 5, 6, 7
    mova                 m7, [o(pixel_10bpc_max)]
    paddw                m0, [dstq+strideq*0]
    paddw                m1, [dstq+strideq*1]
    paddw                m2, [dstq+strideq*2]
    paddw                m3, [dstq+r3]
    REPX     {pminsw x, m7}, m0, m1, m2, m3
    REPX     {pmaxsw x, m4}, m0, m1, m2, m3
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    mova   [dstq+strideq*2], m2
    mova   [dstq+r3       ], m3
    RET

INV_TXFM_8X4_FN adst, dct
INV_TXFM_8X4_FN adst, adst
INV_TXFM_8X4_FN adst, flipadst
INV_TXFM_8X4_FN adst, identity

cglobal iadst_8x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call m(idct_8x4_internal_16bpc).load
    call .main_pass1
    jmp m(idct_8x4_internal_16bpc).pack_transpose
.main_pass1:
%if ARCH_X86_64
    mova                m11, [o(pd_2048)]
    mova                m12, [o(clip_min)]
    mova                m13, [o(clip_max)]

    ITX_MULSUB_2D         7, 0, 8, 9, 10, 11,  401, 4076 ; t1a, t0a
    ITX_MULSUB_2D         1, 6, 8, 9, 10, 11, 3920, 1189 ; t7a, t6a
    ITX_MULSUB_2D         5, 2, 8, 9, 10, 11, 1931, 3612 ; t3a, t2a
    ITX_MULSUB_2D         3, 4, 8, 9, 10, 11, 3166, 2598 ; t5a, t4a
    psubd                m8, m2, m6 ; t6
    paddd                m2, m6     ; t2
    psubd                m6, m0, m4 ; t4
    paddd                m0, m4     ; t0
    psubd                m4, m5, m1 ; t7
    paddd                m5, m1     ; t3
    psubd                m1, m7, m3 ; t5
    paddd                m7, m3     ; t1
    REPX    {pmaxsd x, m12}, m6, m1, m8, m4, m2, m0, m5, m7
    REPX    {pminsd x, m13}, m6, m1, m8, m4, m2, m0, m5, m7
    ITX_MULSUB_2D         6, 1, 3, 9, 10, 11, 1567, 3784 ; t5a, t4a
    ITX_MULSUB_2D         4, 8, 3, 9, 10, 11, 3784, 10   ; t6a, t7a
    psubd                m9, m6, m8 ;  t7
    paddd                m6, m8     ;  out6
    mova                 m8, [o(pd_2896)]
    psubd                m3, m7, m5 ;  t3
    paddd                m7, m5     ; -out7
    psubd                m5, m0, m2 ;  t2
    paddd                m0, m2     ;  out0
    psubd                m2, m1, m4 ;  t6
    paddd                m1, m4     ; -out1
    REPX    {pmaxsd x, m12}, m5, m3, m2, m9
    REPX    {pminsd x, m13}, m5, m3, m2, m9
    REPX    {pmulld x, m8 }, m5, m3, m2, m9
    psubd               m4, m5, m3 ; (t2 - t3) * 2896
    paddd               m3, m5     ; (t2 + t3) * 2896
    psubd               m5, m2, m9 ; (t6 - t7) * 2896
    paddd               m2, m9     ; (t6 + t7) * 2896

    ; m0=out0,m1=-out1,m6=out6,m7=-out7

    pcmpeqd              m8, m8
    REPX     {pxor  x, m8 }, m1, m7, m3, m5
    REPX     {psubd x, m8 }, m1, m7
    REPX     {paddd x, m11}, m2, m3, m4, m5
    REPX     {psrad x, 12 }, m2, m3, m4, m5
%else
    mova [rsp+0*16+2*gprsize], m2
    mova [rsp+1*16+2*gprsize], m3
    mova [rsp+2*16+2*gprsize], m4
    mova [rsp+3*16+2*gprsize], m5
    mova                 m5, [o(pd_2048)]

    ITX_MULSUB_2D         7, 0, 2, 3, 4, 5,  401, 4076 ; t1a, t0a
    ITX_MULSUB_2D         1, 6, 2, 3, 4, 5, 3920, 1189 ; t7a, t6a
    mova                 m2, [rsp+0*16+2*gprsize]
    mova                 m3, [rsp+1*16+2*gprsize]
    mova                 m4, [rsp+2*16+2*gprsize]
    mova [rsp+0*16+2*gprsize], m0
    mova [rsp+1*16+2*gprsize], m1
    mova [rsp+2*16+2*gprsize], m6
    mova                 m1, [rsp+3*16+2*gprsize]
    mova [rsp+3*16+2*gprsize], m7
    ITX_MULSUB_2D         1, 2, 0, 6, 7, 5, 1931, 3612 ; t3a, t2a
    ITX_MULSUB_2D         3, 4, 0, 6, 7, 5, 3166, 2598 ; t5a, t4a
    mova                 m0, [rsp+0*16+2*gprsize]
    mova                 m6, [rsp+2*16+2*gprsize]
    psubd                m7, m2, m6 ; t6
    paddd                m2, m6     ; t2
    psubd                m6, m0, m4 ; t4
    paddd                m0, m4     ; t0
    mova [rsp+0*16+2*gprsize], m7
    mova                 m5, [rsp+1*16+2*gprsize]
    mova                 m7, [rsp+3*16+2*gprsize]
    psubd                m4, m1, m5 ; t7
    paddd                m5, m1     ; t3
    psubd                m1, m7, m3 ; t5
    paddd                m7, m3     ; t1
    mova                 m3, [o(clip_min)]
    REPX    {pmaxsd x, m3 }, m6, m1, m4, m2, m0, m5, m7
    mova [rsp+1*16+2*gprsize], m7
    mova                 m7, [o(clip_max)]
    pmaxsd               m3, [rsp+0*16+2*gprsize]
    REPX    {pminsd x, m7 }, m6, m1, m3, m4, m2, m0, m5
    pminsd               m7, [rsp+1*16+2*gprsize]
    mova [rsp+0*16+2*gprsize], m0
    mova [rsp+1*16+2*gprsize], m2
    mova [rsp+2*16+2*gprsize], m5
    mova [rsp+3*16+2*gprsize], m7
    mova                 m0, [o(pd_2048)]
    ITX_MULSUB_2D         6, 1, 2, 5, 7, 0, 1567, 3784 ; t5a, t4a
    ITX_MULSUB_2D         4, 3, 2, 5, 7, 0, 3784, 7    ; t6a, t7a
    mova                 m5, [rsp+2*16+2*gprsize]
    mova                 m7, [rsp+3*16+2*gprsize]
    psubd                m2, m6, m3 ;  t7
    paddd                m6, m3     ;  out6
    mova [rsp+3*16+2*gprsize], m6
    mova                 m0, [rsp+0*16+2*gprsize]
    mova                 m6, [rsp+1*16+2*gprsize]
    psubd                m3, m7, m5 ;  t3
    paddd                m7, m5     ; -out7
    psubd                m5, m0, m6 ;  t2
    paddd                m0, m6     ;  out0
    psubd                m6, m1, m4 ;  t6
    paddd                m1, m4     ; -out1
    mova                 m4, [o(clip_min)]
    REPX    {pmaxsd x, m4 }, m5, m3, m6, m2
    mova                 m4, [o(clip_max)]
    REPX    {pminsd x, m4 }, m5, m3, m6, m2
    mova                 m4, [o(pd_2896)]
    REPX    {pmulld x, m4 }, m5, m3, m6, m2
    psubd               m4, m5, m3 ; (t2 - t3) * 2896
    paddd               m3, m5     ; (t2 + t3) * 2896
    psubd               m5, m6, m2 ; (t6 - t7) * 2896
    paddd               m2, m6     ; (t6 + t7) * 2896
    mova [rsp+2*16+2*gprsize], m0

    pcmpeqd              m0, m0
    mova                 m6, [o(pd_2048)]
    REPX     {pxor  x, m0 }, m1, m7, m3, m5
    REPX     {psubd x, m0 }, m1, m7
    REPX     {paddd x, m6 }, m2, m3, m4, m5
    REPX     {psrad x, 12 }, m2, m3, m4, m5

    mova                 m6, [rsp+3*16+2*gprsize]
    mova                 m0, [rsp+2*16+2*gprsize]
%endif
    ret

.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_8x4_internal_8bpc, _ssse3).main
    jmp m(idct_8x4_internal_16bpc).end

INV_TXFM_8X4_FN flipadst, dct
INV_TXFM_8X4_FN flipadst, adst
INV_TXFM_8X4_FN flipadst, flipadst
INV_TXFM_8X4_FN flipadst, identity

cglobal iflipadst_8x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call m(idct_8x4_internal_16bpc).load
    call m(iadst_8x4_internal_16bpc).main_pass1
    packssdw             m7, m6
    packssdw             m5, m4
    packssdw             m3, m2
    packssdw             m1, m0
    mova                 m0, m7
    mova                 m2, m5
    mova                 m4, m3
    mova                 m6, m1
    jmp m(idct_8x4_internal_16bpc).transpose
.pass2:
%if ARCH_X86_32
    lea                  r5, [o(itx8_start)]
%endif
    call m_suffix(iadst_8x4_internal_8bpc, _ssse3).main
    lea                  r3, [strideq*3]
    add                dstq, r3
    neg             strideq
    neg                  r3
    jmp m(idct_8x4_internal_16bpc).end2

INV_TXFM_8X4_FN identity, dct
INV_TXFM_8X4_FN identity, adst
INV_TXFM_8X4_FN identity, flipadst
INV_TXFM_8X4_FN identity, identity

cglobal iidentity_8x4_internal_16bpc, 0, 0, 0, dst, stride, c, eob, tx2
    call m(idct_8x4_internal_16bpc).load
    REPX       {paddd x, x}, m0, m1, m2, m3, m4, m5, m6, m7
    jmp m(idct_8x4_internal_16bpc).pack_transpose
.pass2:
    mova                 m7, [o(pw_1697x8)]
    pmulhrsw             m4, m7, m0
    pmulhrsw             m5, m7, m1
    pmulhrsw             m6, m7, m2
    pmulhrsw             m7, m3
    paddsw               m0, m4
    paddsw               m1, m5
    paddsw               m2, m6
    paddsw               m3, m7
    jmp m(idct_8x4_internal_16bpc).end
