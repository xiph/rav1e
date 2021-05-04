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

SECTION_RODATA

subpel_h_shufA: db 0,  1,  2,  3,  2,  3,  4,  5,  4,  5,  6,  7,  6,  7,  8,  9
subpel_h_shufB: db 4,  5,  6,  7,  6,  7,  8,  9,  8,  9, 10, 11, 10, 11, 12, 13
subpel_h_shuf2: db 0,  1,  2,  3,  4,  5,  6,  7,  2,  3,  4,  5,  6,  7,  8,  9

put_bilin_h_rnd:  dw  8,  8, 10, 10
prep_mul:         dw 16, 16,  4,  4
put_8tap_h_rnd:   dd 34, 40
prep_8tap_1d_rnd: dd     8 - (8192 <<  4)
prep_8tap_2d_rnd: dd    32 - (8192 <<  5)

%define pw_16 prep_mul

pw_2:     times 2 dw 2
pw_2048:  times 2 dw 2048
pw_8192:  times 2 dw 8192
pw_32766: times 2 dw 32766
pd_32:    dd 32
pd_512:   dd 512

%macro BASE_JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - %3)
    %xdefine %%base %1_%2
    %%table:
    %rep %0 - 2
        dw %%base %+ _w%3 - %%base
        %rotate 1
    %endrep
%endmacro

%xdefine put_avx2 mangle(private_prefix %+ _put_bilin_16bpc_avx2.put)
%xdefine prep_avx2 mangle(private_prefix %+ _prep_bilin_16bpc_avx2.prep)

BASE_JMP_TABLE put,  avx2, 2, 4, 8, 16, 32, 64, 128
BASE_JMP_TABLE prep, avx2,    4, 8, 16, 32, 64, 128

%macro HV_JMP_TABLE 5-*
    %xdefine %%prefix mangle(private_prefix %+ _%1_%2_16bpc_%3)
    %xdefine %%base %1_%3
    %assign %%types %4
    %if %%types & 1
        %xdefine %1_%2_h_%3_table  (%%h  - %5)
        %%h:
        %rep %0 - 4
            dw %%prefix %+ .h_w%5 - %%base
            %rotate 1
        %endrep
        %rotate 4
    %endif
    %if %%types & 2
        %xdefine %1_%2_v_%3_table  (%%v  - %5)
        %%v:
        %rep %0 - 4
            dw %%prefix %+ .v_w%5 - %%base
            %rotate 1
        %endrep
        %rotate 4
    %endif
    %if %%types & 4
        %xdefine %1_%2_hv_%3_table (%%hv - %5)
        %%hv:
        %rep %0 - 4
            dw %%prefix %+ .hv_w%5 - %%base
            %rotate 1
        %endrep
    %endif
%endmacro

HV_JMP_TABLE put,  bilin, avx2, 7, 2, 4, 8, 16, 32, 64, 128
HV_JMP_TABLE prep, bilin, avx2, 7,    4, 8, 16, 32, 64, 128

%define table_offset(type, fn) type %+ fn %+ SUFFIX %+ _table - type %+ SUFFIX

cextern mc_subpel_filters
%define subpel_filters (mangle(private_prefix %+ _mc_subpel_filters)-8)

SECTION .text

INIT_XMM avx2
cglobal put_bilin_16bpc, 4, 8, 0, dst, ds, src, ss, w, h, mxy
    mov                mxyd, r6m ; mx
    lea                  r7, [put_avx2]
%if UNIX64
    DECLARE_REG_TMP 8
    %define org_w r8d
    mov                 r8d, wd
%else
    DECLARE_REG_TMP 7
    %define org_w wm
%endif
    tzcnt                wd, wm
    movifnidn            hd, hm
    test               mxyd, mxyd
    jnz .h
    mov                mxyd, r7m ; my
    test               mxyd, mxyd
    jnz .v
.put:
    movzx                wd, word [r7+wq*2+table_offset(put,)]
    add                  wq, r7
    jmp                  wq
.put_w2:
    mov                 r6d, [srcq+ssq*0]
    mov                 r7d, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    mov        [dstq+dsq*0], r6d
    mov        [dstq+dsq*1], r7d
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .put_w2
    RET
.put_w4:
    mov                  r6, [srcq+ssq*0]
    mov                  r7, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    mov        [dstq+dsq*0], r6
    mov        [dstq+dsq*1], r7
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .put_w4
    RET
.put_w8:
    movu                 m0, [srcq+ssq*0]
    movu                 m1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    mova       [dstq+dsq*0], m0
    mova       [dstq+dsq*1], m1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .put_w8
    RET
INIT_YMM avx2
.put_w16:
    movu                 m0, [srcq+ssq*0]
    movu                 m1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    mova       [dstq+dsq*0], m0
    mova       [dstq+dsq*1], m1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .put_w16
    RET
.put_w32:
    movu                 m0, [srcq+ssq*0+32*0]
    movu                 m1, [srcq+ssq*0+32*1]
    movu                 m2, [srcq+ssq*1+32*0]
    movu                 m3, [srcq+ssq*1+32*1]
    lea                srcq, [srcq+ssq*2]
    mova  [dstq+dsq*0+32*0], m0
    mova  [dstq+dsq*0+32*1], m1
    mova  [dstq+dsq*1+32*0], m2
    mova  [dstq+dsq*1+32*1], m3
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .put_w32
    RET
.put_w64:
    movu                 m0, [srcq+32*0]
    movu                 m1, [srcq+32*1]
    movu                 m2, [srcq+32*2]
    movu                 m3, [srcq+32*3]
    add                srcq, ssq
    mova        [dstq+32*0], m0
    mova        [dstq+32*1], m1
    mova        [dstq+32*2], m2
    mova        [dstq+32*3], m3
    add                dstq, dsq
    dec                  hd
    jg .put_w64
    RET
.put_w128:
    movu                 m0, [srcq+32*0]
    movu                 m1, [srcq+32*1]
    movu                 m2, [srcq+32*2]
    movu                 m3, [srcq+32*3]
    mova        [dstq+32*0], m0
    mova        [dstq+32*1], m1
    mova        [dstq+32*2], m2
    mova        [dstq+32*3], m3
    movu                 m0, [srcq+32*4]
    movu                 m1, [srcq+32*5]
    movu                 m2, [srcq+32*6]
    movu                 m3, [srcq+32*7]
    add                srcq, ssq
    mova        [dstq+32*4], m0
    mova        [dstq+32*5], m1
    mova        [dstq+32*6], m2
    mova        [dstq+32*7], m3
    add                dstq, dsq
    dec                  hd
    jg .put_w128
    RET
.h:
    movd                xm5, mxyd
    mov                mxyd, r7m ; my
    vpbroadcastd         m4, [pw_16]
    vpbroadcastw         m5, xm5
    psubw                m4, m5
    test               mxyd, mxyd
    jnz .hv
    ; 12-bit is rounded twice so we can't use the same pmulhrsw approach as .v
    movzx                wd, word [r7+wq*2+table_offset(put, _bilin_h)]
    mov                 r6d, r8m ; bitdepth_max
    add                  wq, r7
    shr                 r6d, 11
    vpbroadcastd         m3, [r7-put_avx2+put_bilin_h_rnd+r6*4]
    jmp                  wq
.h_w2:
    movq                xm1, [srcq+ssq*0]
    movhps              xm1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    pmullw              xm0, xm4, xm1
    psrlq               xm1, 16
    pmullw              xm1, xm5
    paddw               xm0, xm3
    paddw               xm0, xm1
    psrlw               xm0, 4
    movd       [dstq+dsq*0], xm0
    pextrd     [dstq+dsq*1], xm0, 2
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w2
    RET
.h_w4:
    movq                xm0, [srcq+ssq*0]
    movhps              xm0, [srcq+ssq*1]
    movq                xm1, [srcq+ssq*0+2]
    movhps              xm1, [srcq+ssq*1+2]
    lea                srcq, [srcq+ssq*2]
    pmullw              xm0, xm4
    pmullw              xm1, xm5
    paddw               xm0, xm3
    paddw               xm0, xm1
    psrlw               xm0, 4
    movq       [dstq+dsq*0], xm0
    movhps     [dstq+dsq*1], xm0
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w4
    RET
.h_w8:
    movu                xm0, [srcq+ssq*0]
    vinserti128          m0, [srcq+ssq*1], 1
    movu                xm1, [srcq+ssq*0+2]
    vinserti128          m1, [srcq+ssq*1+2], 1
    lea                srcq, [srcq+ssq*2]
    pmullw               m0, m4
    pmullw               m1, m5
    paddw                m0, m3
    paddw                m0, m1
    psrlw                m0, 4
    mova         [dstq+dsq*0], xm0
    vextracti128 [dstq+dsq*1], m0, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w8
    RET
.h_w16:
    pmullw               m0, m4, [srcq+ssq*0]
    pmullw               m1, m5, [srcq+ssq*0+2]
    paddw                m0, m3
    paddw                m0, m1
    pmullw               m1, m4, [srcq+ssq*1]
    pmullw               m2, m5, [srcq+ssq*1+2]
    lea                srcq, [srcq+ssq*2]
    paddw                m1, m3
    paddw                m1, m2
    psrlw                m0, 4
    psrlw                m1, 4
    mova       [dstq+dsq*0], m0
    mova       [dstq+dsq*1], m1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w16
    RET
.h_w32:
    pmullw               m0, m4, [srcq+32*0]
    pmullw               m1, m5, [srcq+32*0+2]
    paddw                m0, m3
    paddw                m0, m1
    pmullw               m1, m4, [srcq+32*1]
    pmullw               m2, m5, [srcq+32*1+2]
    add                srcq, ssq
    paddw                m1, m3
    paddw                m1, m2
    psrlw                m0, 4
    psrlw                m1, 4
    mova        [dstq+32*0], m0
    mova        [dstq+32*1], m1
    add                dstq, dsq
    dec                  hd
    jg .h_w32
    RET
.h_w64:
.h_w128:
    movifnidn           t0d, org_w
.h_w64_loop0:
    mov                 r6d, t0d
.h_w64_loop:
    pmullw               m0, m4, [srcq+r6*2-32*1]
    pmullw               m1, m5, [srcq+r6*2-32*1+2]
    paddw                m0, m3
    paddw                m0, m1
    pmullw               m1, m4, [srcq+r6*2-32*2]
    pmullw               m2, m5, [srcq+r6*2-32*2+2]
    paddw                m1, m3
    paddw                m1, m2
    psrlw                m0, 4
    psrlw                m1, 4
    mova   [dstq+r6*2-32*1], m0
    mova   [dstq+r6*2-32*2], m1
    sub                 r6d, 32
    jg .h_w64_loop
    add                srcq, ssq
    add                dstq, dsq
    dec                  hd
    jg .h_w64_loop0
    RET
.v:
    movzx                wd, word [r7+wq*2+table_offset(put, _bilin_v)]
    shl                mxyd, 11
    movd                xm5, mxyd
    add                  wq, r7
    vpbroadcastw         m5, xm5
    jmp                  wq
.v_w2:
    movd                xm0, [srcq+ssq*0]
.v_w2_loop:
    movd                xm1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    punpckldq           xm2, xm0, xm1
    movd                xm0, [srcq+ssq*0]
    punpckldq           xm1, xm0
    psubw               xm1, xm2
    pmulhrsw            xm1, xm5
    paddw               xm1, xm2
    movd       [dstq+dsq*0], xm1
    pextrd     [dstq+dsq*1], xm1, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w2_loop
    RET
.v_w4:
    movq                xm0, [srcq+ssq*0]
.v_w4_loop:
    movq                xm1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    punpcklqdq          xm2, xm0, xm1
    movq                xm0, [srcq+ssq*0]
    punpcklqdq          xm1, xm0
    psubw               xm1, xm2
    pmulhrsw            xm1, xm5
    paddw               xm1, xm2
    movq       [dstq+dsq*0], xm1
    movhps     [dstq+dsq*1], xm1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w4_loop
    RET
.v_w8:
    movu                xm0, [srcq+ssq*0]
.v_w8_loop:
    vbroadcasti128       m1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    vpblendd             m2, m0, m1, 0xf0
    vbroadcasti128       m0, [srcq+ssq*0]
    vpblendd             m1, m0, 0xf0
    psubw                m1, m2
    pmulhrsw             m1, m5
    paddw                m1, m2
    mova         [dstq+dsq*0], xm1
    vextracti128 [dstq+dsq*1], m1, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w8_loop
    RET
.v_w32:
    movu                 m0, [srcq+ssq*0+32*0]
    movu                 m1, [srcq+ssq*0+32*1]
.v_w32_loop:
    movu                 m2, [srcq+ssq*1+32*0]
    movu                 m3, [srcq+ssq*1+32*1]
    lea                srcq, [srcq+ssq*2]
    psubw                m4, m2, m0
    pmulhrsw             m4, m5
    paddw                m4, m0
    movu                 m0, [srcq+ssq*0+32*0]
    mova  [dstq+dsq*0+32*0], m4
    psubw                m4, m3, m1
    pmulhrsw             m4, m5
    paddw                m4, m1
    movu                 m1, [srcq+ssq*0+32*1]
    mova  [dstq+dsq*0+32*1], m4
    psubw                m4, m0, m2
    pmulhrsw             m4, m5
    paddw                m4, m2
    mova  [dstq+dsq*1+32*0], m4
    psubw                m4, m1, m3
    pmulhrsw             m4, m5
    paddw                m4, m3
    mova  [dstq+dsq*1+32*1], m4
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w32_loop
    RET
.v_w16:
.v_w64:
.v_w128:
    movifnidn           t0d, org_w
    add                 t0d, t0d
    mov                  r4, srcq
    lea                 r6d, [hq+t0*8-256]
    mov                  r7, dstq
.v_w16_loop0:
    movu                 m0, [srcq+ssq*0]
.v_w16_loop:
    movu                 m3, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    psubw                m1, m3, m0
    pmulhrsw             m1, m5
    paddw                m1, m0
    movu                 m0, [srcq+ssq*0]
    psubw                m2, m0, m3
    pmulhrsw             m2, m5
    paddw                m2, m3
    mova       [dstq+dsq*0], m1
    mova       [dstq+dsq*1], m2
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w16_loop
    add                  r4, 32
    add                  r7, 32
    movzx                hd, r6b
    mov                srcq, r4
    mov                dstq, r7
    sub                 r6d, 1<<8
    jg .v_w16_loop0
    RET
.hv:
    movzx                wd, word [r7+wq*2+table_offset(put, _bilin_hv)]
    WIN64_SPILL_XMM       8
    shl                mxyd, 11
    vpbroadcastd         m3, [pw_2]
    movd                xm6, mxyd
    vpbroadcastd         m7, [pw_8192]
    add                  wq, r7
    vpbroadcastw         m6, xm6
    test          dword r8m, 0x800
    jnz .hv_12bpc
    psllw                m4, 2
    psllw                m5, 2
    vpbroadcastd         m7, [pw_2048]
.hv_12bpc:
    jmp                  wq
.hv_w2:
    vpbroadcastq        xm1, [srcq+ssq*0]
    pmullw              xm0, xm4, xm1
    psrlq               xm1, 16
    pmullw              xm1, xm5
    paddw               xm0, xm3
    paddw               xm0, xm1
    psrlw               xm0, 2
.hv_w2_loop:
    movq                xm2, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    movhps              xm2, [srcq+ssq*0]
    pmullw              xm1, xm4, xm2
    psrlq               xm2, 16
    pmullw              xm2, xm5
    paddw               xm1, xm3
    paddw               xm1, xm2
    psrlw               xm1, 2              ; 1 _ 2 _
    shufpd              xm2, xm0, xm1, 0x01 ; 0 _ 1 _
    mova                xm0, xm1
    psubw               xm1, xm2
    paddw               xm1, xm1
    pmulhw              xm1, xm6
    paddw               xm1, xm2
    pmulhrsw            xm1, xm7
    movd       [dstq+dsq*0], xm1
    pextrd     [dstq+dsq*1], xm1, 2
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w2_loop
    RET
.hv_w4:
    pmullw              xm0, xm4, [srcq+ssq*0-8]
    pmullw              xm1, xm5, [srcq+ssq*0-6]
    paddw               xm0, xm3
    paddw               xm0, xm1
    psrlw               xm0, 2
.hv_w4_loop:
    movq                xm1, [srcq+ssq*1]
    movq                xm2, [srcq+ssq*1+2]
    lea                srcq, [srcq+ssq*2]
    movhps              xm1, [srcq+ssq*0]
    movhps              xm2, [srcq+ssq*0+2]
    pmullw              xm1, xm4
    pmullw              xm2, xm5
    paddw               xm1, xm3
    paddw               xm1, xm2
    psrlw               xm1, 2              ; 1 2
    shufpd              xm2, xm0, xm1, 0x01 ; 0 1
    mova                xm0, xm1
    psubw               xm1, xm2
    paddw               xm1, xm1
    pmulhw              xm1, xm6
    paddw               xm1, xm2
    pmulhrsw            xm1, xm7
    movq       [dstq+dsq*0], xm1
    movhps     [dstq+dsq*1], xm1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w4_loop
    RET
.hv_w8:
    pmullw              xm0, xm4, [srcq+ssq*0]
    pmullw              xm1, xm5, [srcq+ssq*0+2]
    paddw               xm0, xm3
    paddw               xm0, xm1
    psrlw               xm0, 2
    vinserti128          m0, xm0, 1
.hv_w8_loop:
    movu                xm1, [srcq+ssq*1]
    movu                xm2, [srcq+ssq*1+2]
    lea                srcq, [srcq+ssq*2]
    vinserti128          m1, [srcq+ssq*0], 1
    vinserti128          m2, [srcq+ssq*0+2], 1
    pmullw               m1, m4
    pmullw               m2, m5
    paddw                m1, m3
    paddw                m1, m2
    psrlw                m1, 2            ; 1 2
    vperm2i128           m2, m0, m1, 0x21 ; 0 1
    mova                 m0, m1
    psubw                m1, m2
    paddw                m1, m1
    pmulhw               m1, m6
    paddw                m1, m2
    pmulhrsw             m1, m7
    mova         [dstq+dsq*0], xm1
    vextracti128 [dstq+dsq*1], m1, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w8_loop
    RET
.hv_w16:
.hv_w32:
.hv_w64:
.hv_w128:
%if UNIX64
    lea                 r6d, [r8*2-32]
%else
    mov                 r6d, wm
    lea                 r6d, [r6*2-32]
%endif
    mov                  r4, srcq
    lea                 r6d, [hq+r6*8]
    mov                  r7, dstq
.hv_w16_loop0:
    pmullw               m0, m4, [srcq+ssq*0]
    pmullw               m1, m5, [srcq+ssq*0+2]
    paddw                m0, m3
    paddw                m0, m1
    psrlw                m0, 2
.hv_w16_loop:
    pmullw               m1, m4, [srcq+ssq*1]
    pmullw               m2, m5, [srcq+ssq*1+2]
    lea                srcq, [srcq+ssq*2]
    paddw                m1, m3
    paddw                m1, m2
    psrlw                m1, 2
    psubw                m2, m1, m0
    paddw                m2, m2
    pmulhw               m2, m6
    paddw                m2, m0
    pmulhrsw             m2, m7
    mova       [dstq+dsq*0], m2
    pmullw               m0, m4, [srcq+ssq*0]
    pmullw               m2, m5, [srcq+ssq*0+2]
    paddw                m0, m3
    paddw                m0, m2
    psrlw                m0, 2
    psubw                m2, m0, m1
    paddw                m2, m2
    pmulhw               m2, m6
    paddw                m2, m1
    pmulhrsw             m2, m7
    mova       [dstq+dsq*1], m2
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w16_loop
    add                  r4, 32
    add                  r7, 32
    movzx                hd, r6b
    mov                srcq, r4
    mov                dstq, r7
    sub                 r6d, 1<<8
    jg .hv_w16_loop0
    RET

cglobal prep_bilin_16bpc, 3, 7, 0, tmp, src, stride, w, h, mxy, stride3
    movifnidn          mxyd, r5m ; mx
    lea                  r6, [prep_avx2]
%if UNIX64
    DECLARE_REG_TMP 7
    %define org_w r7d
%else
    DECLARE_REG_TMP 6
    %define org_w r5m
%endif
    mov               org_w, wd
    tzcnt                wd, wm
    movifnidn            hd, hm
    test               mxyd, mxyd
    jnz .h
    mov                mxyd, r6m ; my
    test               mxyd, mxyd
    jnz .v
.prep:
    movzx                wd, word [r6+wq*2+table_offset(prep,)]
    mov                 r5d, r7m ; bitdepth_max
    vpbroadcastd         m5, [r6-prep_avx2+pw_8192]
    add                  wq, r6
    shr                 r5d, 11
    vpbroadcastd         m4, [r6-prep_avx2+prep_mul+r5*4]
    lea            stride3q, [strideq*3]
    jmp                  wq
.prep_w4:
    movq                xm0, [srcq+strideq*0]
    movhps              xm0, [srcq+strideq*1]
    vpbroadcastq         m1, [srcq+strideq*2]
    vpbroadcastq         m2, [srcq+stride3q ]
    lea                srcq, [srcq+strideq*4]
    vpblendd             m0, m1, 0x30
    vpblendd             m0, m2, 0xc0
    pmullw               m0, m4
    psubw                m0, m5
    mova             [tmpq], m0
    add                tmpq, 32
    sub                  hd, 4
    jg .prep_w4
    RET
.prep_w8:
    movu                xm0, [srcq+strideq*0]
    vinserti128          m0, [srcq+strideq*1], 1
    movu                xm1, [srcq+strideq*2]
    vinserti128          m1, [srcq+stride3q ], 1
    lea                srcq, [srcq+strideq*4]
    pmullw               m0, m4
    pmullw               m1, m4
    psubw                m0, m5
    psubw                m1, m5
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    add                tmpq, 32*2
    sub                  hd, 4
    jg .prep_w8
    RET
.prep_w16:
    pmullw               m0, m4, [srcq+strideq*0]
    pmullw               m1, m4, [srcq+strideq*1]
    pmullw               m2, m4, [srcq+strideq*2]
    pmullw               m3, m4, [srcq+stride3q ]
    lea                srcq, [srcq+strideq*4]
    psubw                m0, m5
    psubw                m1, m5
    psubw                m2, m5
    psubw                m3, m5
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    mova        [tmpq+32*2], m2
    mova        [tmpq+32*3], m3
    add                tmpq, 32*4
    sub                  hd, 4
    jg .prep_w16
    RET
.prep_w32:
    pmullw               m0, m4, [srcq+strideq*0+32*0]
    pmullw               m1, m4, [srcq+strideq*0+32*1]
    pmullw               m2, m4, [srcq+strideq*1+32*0]
    pmullw               m3, m4, [srcq+strideq*1+32*1]
    lea                srcq, [srcq+strideq*2]
    psubw                m0, m5
    psubw                m1, m5
    psubw                m2, m5
    psubw                m3, m5
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    mova        [tmpq+32*2], m2
    mova        [tmpq+32*3], m3
    add                tmpq, 32*4
    sub                  hd, 2
    jg .prep_w32
    RET
.prep_w64:
    pmullw               m0, m4, [srcq+32*0]
    pmullw               m1, m4, [srcq+32*1]
    pmullw               m2, m4, [srcq+32*2]
    pmullw               m3, m4, [srcq+32*3]
    add                srcq, strideq
    psubw                m0, m5
    psubw                m1, m5
    psubw                m2, m5
    psubw                m3, m5
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    mova        [tmpq+32*2], m2
    mova        [tmpq+32*3], m3
    add                tmpq, 32*4
    dec                  hd
    jg .prep_w64
    RET
.prep_w128:
    pmullw               m0, m4, [srcq+32*0]
    pmullw               m1, m4, [srcq+32*1]
    pmullw               m2, m4, [srcq+32*2]
    pmullw               m3, m4, [srcq+32*3]
    psubw                m0, m5
    psubw                m1, m5
    psubw                m2, m5
    psubw                m3, m5
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    mova        [tmpq+32*2], m2
    mova        [tmpq+32*3], m3
    pmullw               m0, m4, [srcq+32*4]
    pmullw               m1, m4, [srcq+32*5]
    pmullw               m2, m4, [srcq+32*6]
    pmullw               m3, m4, [srcq+32*7]
    add                tmpq, 32*8
    add                srcq, strideq
    psubw                m0, m5
    psubw                m1, m5
    psubw                m2, m5
    psubw                m3, m5
    mova        [tmpq-32*4], m0
    mova        [tmpq-32*3], m1
    mova        [tmpq-32*2], m2
    mova        [tmpq-32*1], m3
    dec                  hd
    jg .prep_w128
    RET
.h:
    movd                xm5, mxyd
    mov                mxyd, r6m ; my
    vpbroadcastd         m4, [pw_16]
    vpbroadcastw         m5, xm5
    vpbroadcastd         m3, [pw_32766]
    psubw                m4, m5
    test          dword r7m, 0x800
    jnz .h_12bpc
    psllw                m4, 2
    psllw                m5, 2
.h_12bpc:
    test               mxyd, mxyd
    jnz .hv
    movzx                wd, word [r6+wq*2+table_offset(prep, _bilin_h)]
    add                  wq, r6
    lea            stride3q, [strideq*3]
    jmp                  wq
.h_w4:
    movu                xm1, [srcq+strideq*0]
    vinserti128          m1, [srcq+strideq*2], 1
    movu                xm2, [srcq+strideq*1]
    vinserti128          m2, [srcq+stride3q ], 1
    lea                srcq, [srcq+strideq*4]
    punpcklqdq           m0, m1, m2
    psrldq               m1, 2
    pslldq               m2, 6
    pmullw               m0, m4
    vpblendd             m1, m2, 0xcc
    pmullw               m1, m5
    psubw                m0, m3
    paddw                m0, m1
    psraw                m0, 2
    mova             [tmpq], m0
    add                tmpq, 32
    sub                  hd, 4
    jg .h_w4
    RET
.h_w8:
    movu                xm0, [srcq+strideq*0]
    vinserti128          m0, [srcq+strideq*1], 1
    movu                xm1, [srcq+strideq*0+2]
    vinserti128          m1, [srcq+strideq*1+2], 1
    lea                srcq, [srcq+strideq*2]
    pmullw               m0, m4
    pmullw               m1, m5
    psubw                m0, m3
    paddw                m0, m1
    psraw                m0, 2
    mova             [tmpq], m0
    add                tmpq, 32
    sub                  hd, 2
    jg .h_w8
    RET
.h_w16:
    pmullw               m0, m4, [srcq+strideq*0]
    pmullw               m1, m5, [srcq+strideq*0+2]
    psubw                m0, m3
    paddw                m0, m1
    pmullw               m1, m4, [srcq+strideq*1]
    pmullw               m2, m5, [srcq+strideq*1+2]
    lea                srcq, [srcq+strideq*2]
    psubw                m1, m3
    paddw                m1, m2
    psraw                m0, 2
    psraw                m1, 2
    mova        [tmpq+32*0], m0
    mova        [tmpq+32*1], m1
    add                tmpq, 32*2
    sub                  hd, 2
    jg .h_w16
    RET
.h_w32:
.h_w64:
.h_w128:
    movifnidn           t0d, org_w
.h_w32_loop0:
    mov                 r3d, t0d
.h_w32_loop:
    pmullw               m0, m4, [srcq+r3*2-32*1]
    pmullw               m1, m5, [srcq+r3*2-32*1+2]
    psubw                m0, m3
    paddw                m0, m1
    pmullw               m1, m4, [srcq+r3*2-32*2]
    pmullw               m2, m5, [srcq+r3*2-32*2+2]
    psubw                m1, m3
    paddw                m1, m2
    psraw                m0, 2
    psraw                m1, 2
    mova   [tmpq+r3*2-32*1], m0
    mova   [tmpq+r3*2-32*2], m1
    sub                 r3d, 32
    jg .h_w32_loop
    add                srcq, strideq
    lea                tmpq, [tmpq+t0*2]
    dec                  hd
    jg .h_w32_loop0
    RET
.v:
    movzx                wd, word [r6+wq*2+table_offset(prep, _bilin_v)]
    movd                xm5, mxyd
    vpbroadcastd         m4, [pw_16]
    vpbroadcastw         m5, xm5
    vpbroadcastd         m3, [pw_32766]
    add                  wq, r6
    lea            stride3q, [strideq*3]
    psubw                m4, m5
    test          dword r7m, 0x800
    jnz .v_12bpc
    psllw                m4, 2
    psllw                m5, 2
.v_12bpc:
    jmp                  wq
.v_w4:
    movq                xm0, [srcq+strideq*0]
.v_w4_loop:
    vpbroadcastq         m2, [srcq+strideq*2]
    vpbroadcastq        xm1, [srcq+strideq*1]
    vpblendd             m2, m0, 0x03 ; 0 2 2 2
    vpbroadcastq         m0, [srcq+stride3q ]
    lea                srcq, [srcq+strideq*4]
    vpblendd             m1, m0, 0xf0 ; 1 1 3 3
    vpbroadcastq         m0, [srcq+strideq*0]
    vpblendd             m1, m2, 0x33 ; 0 1 2 3
    vpblendd             m0, m2, 0x0c ; 4 2 4 4
    punpckhqdq           m2, m1, m0   ; 1 2 3 4
    pmullw               m1, m4
    pmullw               m2, m5
    psubw                m1, m3
    paddw                m1, m2
    psraw                m1, 2
    mova             [tmpq], m1
    add                tmpq, 32
    sub                  hd, 4
    jg .v_w4_loop
    RET
.v_w8:
    movu                xm0, [srcq+strideq*0]
.v_w8_loop:
    vbroadcasti128       m2, [srcq+strideq*1]
    lea                srcq, [srcq+strideq*2]
    vpblendd             m1, m0, m2, 0xf0 ; 0 1
    vbroadcasti128       m0, [srcq+strideq*0]
    vpblendd             m2, m0, 0xf0     ; 1 2
    pmullw               m1, m4
    pmullw               m2, m5
    psubw                m1, m3
    paddw                m1, m2
    psraw                m1, 2
    mova             [tmpq], m1
    add                tmpq, 32
    sub                  hd, 2
    jg .v_w8_loop
    RET
.v_w16:
    movu                 m0, [srcq+strideq*0]
.v_w16_loop:
    movu                 m2, [srcq+strideq*1]
    lea                srcq, [srcq+strideq*2]
    pmullw               m0, m4
    pmullw               m1, m5, m2
    psubw                m0, m3
    paddw                m1, m0
    movu                 m0, [srcq+strideq*0]
    psraw                m1, 2
    pmullw               m2, m4
    mova        [tmpq+32*0], m1
    pmullw               m1, m5, m0
    psubw                m2, m3
    paddw                m1, m2
    psraw                m1, 2
    mova        [tmpq+32*1], m1
    add                tmpq, 32*2
    sub                  hd, 2
    jg .v_w16_loop
    RET
.v_w32:
.v_w64:
.v_w128:
%if WIN64
    PUSH                 r7
%endif
    movifnidn           r7d, org_w
    add                 r7d, r7d
    mov                  r3, srcq
    lea                 r6d, [hq+r7*8-256]
    mov                  r5, tmpq
.v_w32_loop0:
    movu                 m0, [srcq+strideq*0]
.v_w32_loop:
    movu                 m2, [srcq+strideq*1]
    lea                srcq, [srcq+strideq*2]
    pmullw               m0, m4
    pmullw               m1, m5, m2
    psubw                m0, m3
    paddw                m1, m0
    movu                 m0, [srcq+strideq*0]
    psraw                m1, 2
    pmullw               m2, m4
    mova        [tmpq+r7*0], m1
    pmullw               m1, m5, m0
    psubw                m2, m3
    paddw                m1, m2
    psraw                m1, 2
    mova        [tmpq+r7*1], m1
    lea                tmpq, [tmpq+r7*2]
    sub                  hd, 2
    jg .v_w32_loop
    add                  r3, 32
    add                  r5, 32
    movzx                hd, r6b
    mov                srcq, r3
    mov                tmpq, r5
    sub                 r6d, 1<<8
    jg .v_w32_loop0
%if WIN64
    POP                  r7
%endif
    RET
.hv:
    WIN64_SPILL_XMM       7
    movzx                wd, word [r6+wq*2+table_offset(prep, _bilin_hv)]
    shl                mxyd, 11
    movd                xm6, mxyd
    add                  wq, r6
    lea            stride3q, [strideq*3]
    vpbroadcastw         m6, xm6
    jmp                  wq
.hv_w4:
    movu                xm1, [srcq+strideq*0]
%if WIN64
    movaps         [rsp+24], xmm7
%endif
    pmullw              xm0, xm4, xm1
    psrldq              xm1, 2
    pmullw              xm1, xm5
    psubw               xm0, xm3
    paddw               xm0, xm1
    psraw               xm0, 2
    vpbroadcastq         m0, xm0
.hv_w4_loop:
    movu                xm1, [srcq+strideq*1]
    vinserti128          m1, [srcq+stride3q ], 1
    movu                xm2, [srcq+strideq*2]
    lea                srcq, [srcq+strideq*4]
    vinserti128          m2, [srcq+strideq*0], 1
    punpcklqdq           m7, m1, m2
    psrldq               m1, 2
    pslldq               m2, 6
    pmullw               m7, m4
    vpblendd             m1, m2, 0xcc
    pmullw               m1, m5
    psubw                m7, m3
    paddw                m1, m7
    psraw                m1, 2         ; 1 2 3 4
    vpblendd             m0, m1, 0x3f
    vpermq               m2, m0, q2103 ; 0 1 2 3
    mova                 m0, m1
    psubw                m1, m2
    pmulhrsw             m1, m6
    paddw                m1, m2
    mova             [tmpq], m1
    add                tmpq, 32
    sub                  hd, 4
    jg .hv_w4_loop
%if WIN64
    movaps             xmm7, [rsp+24]
%endif
    RET
.hv_w8:
    pmullw              xm0, xm4, [srcq+strideq*0]
    pmullw              xm1, xm5, [srcq+strideq*0+2]
    psubw               xm0, xm3
    paddw               xm0, xm1
    psraw               xm0, 2
    vinserti128          m0, xm0, 1
.hv_w8_loop:
    movu                xm1, [srcq+strideq*1]
    movu                xm2, [srcq+strideq*1+2]
    lea                srcq, [srcq+strideq*2]
    vinserti128          m1, [srcq+strideq*0], 1
    vinserti128          m2, [srcq+strideq*0+2], 1
    pmullw               m1, m4
    pmullw               m2, m5
    psubw                m1, m3
    paddw                m1, m2
    psraw                m1, 2            ; 1 2
    vperm2i128           m2, m0, m1, 0x21 ; 0 1
    mova                 m0, m1
    psubw                m1, m2
    pmulhrsw             m1, m6
    paddw                m1, m2
    mova             [tmpq], m1
    add                tmpq, 32
    sub                  hd, 2
    jg .hv_w8_loop
    RET
.hv_w16:
.hv_w32:
.hv_w64:
.hv_w128:
%if WIN64
    PUSH                 r7
%endif
    movifnidn           r7d, org_w
    add                 r7d, r7d
    mov                  r3, srcq
    lea                 r6d, [hq+r7*8-256]
    mov                  r5, tmpq
.hv_w16_loop0:
    pmullw               m0, m4, [srcq]
    pmullw               m1, m5, [srcq+2]
    psubw                m0, m3
    paddw                m0, m1
    psraw                m0, 2
.hv_w16_loop:
    pmullw               m1, m4, [srcq+strideq*1]
    pmullw               m2, m5, [srcq+strideq*1+2]
    lea                srcq, [srcq+strideq*2]
    psubw                m1, m3
    paddw                m1, m2
    psraw                m1, 2
    psubw                m2, m1, m0
    pmulhrsw             m2, m6
    paddw                m2, m0
    mova        [tmpq+r7*0], m2
    pmullw               m0, m4, [srcq+strideq*0]
    pmullw               m2, m5, [srcq+strideq*0+2]
    psubw                m0, m3
    paddw                m0, m2
    psraw                m0, 2
    psubw                m2, m0, m1
    pmulhrsw             m2, m6
    paddw                m2, m1
    mova        [tmpq+r7*1], m2
    lea                tmpq, [tmpq+r7*2]
    sub                  hd, 2
    jg .hv_w16_loop
    add                  r3, 32
    add                  r5, 32
    movzx                hd, r6b
    mov                srcq, r3
    mov                tmpq, r5
    sub                 r6d, 1<<8
    jg .hv_w16_loop0
%if WIN64
    POP                  r7
%endif
    RET

; int8_t subpel_filters[5][15][8]
%assign FILTER_REGULAR (0*15 << 16) | 3*15
%assign FILTER_SMOOTH  (1*15 << 16) | 4*15
%assign FILTER_SHARP   (2*15 << 16) | 3*15

%macro MC_8TAP_FN 4 ; prefix, type, type_h, type_v
cglobal %1_8tap_%2_16bpc
    mov                 t0d, FILTER_%3
%ifidn %3, %4
    mov                 t1d, t0d
%else
    mov                 t1d, FILTER_%4
%endif
%ifnidn %2, regular ; skip the jump in the last filter
    jmp mangle(private_prefix %+ _%1_8tap_16bpc %+ SUFFIX)
%endif
%endmacro

%if WIN64
DECLARE_REG_TMP 4, 5
%else
DECLARE_REG_TMP 7, 8
%endif

MC_8TAP_FN put, sharp,          SHARP,   SHARP
MC_8TAP_FN put, sharp_smooth,   SHARP,   SMOOTH
MC_8TAP_FN put, smooth_sharp,   SMOOTH,  SHARP
MC_8TAP_FN put, smooth,         SMOOTH,  SMOOTH
MC_8TAP_FN put, sharp_regular,  SHARP,   REGULAR
MC_8TAP_FN put, regular_sharp,  REGULAR, SHARP
MC_8TAP_FN put, smooth_regular, SMOOTH,  REGULAR
MC_8TAP_FN put, regular_smooth, REGULAR, SMOOTH
MC_8TAP_FN put, regular,        REGULAR, REGULAR

cglobal put_8tap_16bpc, 4, 9, 0, dst, ds, src, ss, w, h, mx, my
%define base r8-put_avx2
    imul                mxd, mxm, 0x010101
    add                 mxd, t0d ; 8tap_h, mx, 4tap_h
    imul                myd, mym, 0x010101
    add                 myd, t1d ; 8tap_v, my, 4tap_v
    lea                  r8, [put_avx2]
    movifnidn            wd, wm
    movifnidn            hd, hm
    test                mxd, 0xf00
    jnz .h
    test                myd, 0xf00
    jnz .v
    tzcnt                wd, wd
    movzx                wd, word [r8+wq*2+table_offset(put,)]
    add                  wq, r8
%if WIN64
    pop                  r8
%endif
    jmp                  wq
.h_w2:
    movzx               mxd, mxb
    sub                srcq, 2
    mova                xm2, [subpel_h_shuf2]
    vpbroadcastd        xm3, [base+subpel_filters+mxq*8+2]
    pmovsxbw            xm3, xm3
.h_w2_loop:
    movu                xm0, [srcq+ssq*0]
    movu                xm1, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    pshufb              xm0, xm2
    pshufb              xm1, xm2
    pmaddwd             xm0, xm3
    pmaddwd             xm1, xm3
    phaddd              xm0, xm1
    paddd               xm0, xm4
    psrad               xm0, 6
    packusdw            xm0, xm0
    pminsw              xm0, xm5
    movd       [dstq+dsq*0], xm0
    pextrd     [dstq+dsq*1], xm0, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w2_loop
    RET
.h_w4:
    movzx               mxd, mxb
    sub                srcq, 2
    pmovsxbw            xm3, [base+subpel_filters+mxq*8]
    WIN64_SPILL_XMM       8
    vbroadcasti128       m6, [subpel_h_shufA]
    vbroadcasti128       m7, [subpel_h_shufB]
    pshufd              xm3, xm3, q2211
    vpbroadcastq         m2, xm3
    vpermq               m3, m3, q1111
.h_w4_loop:
    movu                xm1, [srcq+ssq*0]
    vinserti128          m1, [srcq+ssq*1], 1
    lea                srcq, [srcq+ssq*2]
    pshufb               m0, m1, m6 ; 0 1 1 2 2 3 3 4
    pshufb               m1, m7     ; 2 3 3 4 4 5 5 6
    pmaddwd              m0, m2
    pmaddwd              m1, m3
    paddd                m0, m4
    paddd                m0, m1
    psrad                m0, 6
    vextracti128        xm1, m0, 1
    packusdw            xm0, xm1
    pminsw              xm0, xm5
    movq       [dstq+dsq*0], xm0
    movhps     [dstq+dsq*1], xm0
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w4_loop
    RET
.h:
    test                myd, 0xf00
    jnz .hv
    mov                 r7d, r8m
    vpbroadcastw         m5, r8m
    shr                 r7d, 11
    vpbroadcastd         m4, [base+put_8tap_h_rnd+r7*4]
    cmp                  wd, 4
    je .h_w4
    jl .h_w2
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      13
    shr                 mxd, 16
    sub                srcq, 6
    vpbroadcastq         m0, [base+subpel_filters+mxq*8]
    vbroadcasti128       m6, [subpel_h_shufA]
    vbroadcasti128       m7, [subpel_h_shufB]
    punpcklbw            m0, m0
    psraw                m0, 8 ; sign-extend
    pshufd               m8, m0, q0000
    pshufd               m9, m0, q1111
    pshufd              m10, m0, q2222
    pshufd              m11, m0, q3333
    cmp                  wd, 8
    jg .h_w16
.h_w8:
%macro PUT_8TAP_H 5 ; dst/src+0, src+8, src+16, tmp[1-2]
    pshufb              m%4, m%1, m7   ; 2 3 3 4 4 5 5 6
    pshufb              m%1, m6        ; 0 1 1 2 2 3 3 4
    pmaddwd             m%5, m9, m%4   ; abcd1
    pmaddwd             m%1, m8        ; abcd0
    pshufb              m%2, m7        ; 6 7 7 8 8 9 9 a
    shufpd              m%4, m%2, 0x05 ; 4 5 5 6 6 7 7 8
    paddd               m%5, m4
    paddd               m%1, m%5
    pmaddwd             m%5, m11, m%2  ; abcd3
    paddd               m%1, m%5
    pmaddwd             m%5, m10, m%4  ; abcd2
    pshufb              m%3, m7        ; a b b c c d d e
    pmaddwd             m%4, m8        ; efgh0
    paddd               m%1, m%5
    pmaddwd             m%5, m9, m%2   ; efgh1
    shufpd              m%2, m%3, 0x05 ; 8 9 9 a a b b c
    pmaddwd             m%3, m11       ; efgh3
    pmaddwd             m%2, m10       ; efgh2
    paddd               m%4, m4
    paddd               m%4, m%5
    paddd               m%3, m%4
    paddd               m%2, m%3
    psrad               m%1, 6
    psrad               m%2, 6
    packusdw            m%1, m%2
    pminsw              m%1, m5
%endmacro
    movu                xm0, [srcq+ssq*0+ 0]
    vinserti128          m0, [srcq+ssq*1+ 0], 1
    movu                xm2, [srcq+ssq*0+16]
    vinserti128          m2, [srcq+ssq*1+16], 1
    lea                srcq, [srcq+ssq*2]
    shufpd               m1, m0, m2, 0x05
    PUT_8TAP_H            0, 1, 2, 3, 12
    mova         [dstq+dsq*0], xm0
    vextracti128 [dstq+dsq*1], m0, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .h_w8
    RET
.h_w16:
    mov                 r6d, wd
.h_w16_loop:
    movu                 m0, [srcq+r6*2-32]
    movu                 m1, [srcq+r6*2-24]
    movu                 m2, [srcq+r6*2-16]
    PUT_8TAP_H            0, 1, 2, 3, 12
    mova     [dstq+r6*2-32], m0
    sub                 r6d, 16
    jg .h_w16_loop
    add                srcq, ssq
    add                dstq, dsq
    dec                  hd
    jg .h_w16
    RET
.v:
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    vpbroadcastq         m0, [base+subpel_filters+myq*8]
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      15
    vpbroadcastd         m6, [pd_32]
    vpbroadcastw         m7, r8m
    lea                  r6, [ssq*3]
    sub                srcq, r6
    punpcklbw            m0, m0
    psraw                m0, 8 ; sign-extend
    pshufd               m8, m0, q0000
    pshufd               m9, m0, q1111
    pshufd              m10, m0, q2222
    pshufd              m11, m0, q3333
    cmp                  wd, 4
    jg .v_w8
    je .v_w4
.v_w2:
    movd                xm2, [srcq+ssq*0]
    pinsrd              xm2, [srcq+ssq*1], 1
    pinsrd              xm2, [srcq+ssq*2], 2
    pinsrd              xm2, [srcq+r6   ], 3 ; 0 1 2 3
    lea                srcq, [srcq+ssq*4]
    movd                xm3, [srcq+ssq*0]
    vpbroadcastd        xm1, [srcq+ssq*1]
    vpbroadcastd        xm0, [srcq+ssq*2]
    add                srcq, r6
    vpblendd            xm3, xm1, 0x02       ; 4 5
    vpblendd            xm1, xm0, 0x02       ; 5 6
    palignr             xm4, xm3, xm2, 4     ; 1 2 3 4
    punpcklwd           xm3, xm1             ; 45 56
    punpcklwd           xm1, xm2, xm4        ; 01 12
    punpckhwd           xm2, xm4             ; 23 34
.v_w2_loop:
    vpbroadcastd        xm4, [srcq+ssq*0]
    pmaddwd             xm5, xm8, xm1        ; a0 b0
    mova                xm1, xm2
    pmaddwd             xm2, xm9             ; a1 b1
    paddd               xm5, xm6
    paddd               xm5, xm2
    mova                xm2, xm3
    pmaddwd             xm3, xm10            ; a2 b2
    paddd               xm5, xm3
    vpblendd            xm3, xm0, xm4, 0x02  ; 6 7
    vpbroadcastd        xm0, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    vpblendd            xm4, xm0, 0x02       ; 7 8
    punpcklwd           xm3, xm4             ; 67 78
    pmaddwd             xm4, xm11, xm3       ; a3 b3
    paddd               xm5, xm4
    psrad               xm5, 6
    packusdw            xm5, xm5
    pminsw              xm5, xm7
    movd       [dstq+dsq*0], xm5
    pextrd     [dstq+dsq*1], xm5, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w2_loop
    RET
.v_w4:
    movq                xm1, [srcq+ssq*0]
    vpbroadcastq         m0, [srcq+ssq*1]
    vpbroadcastq         m2, [srcq+ssq*2]
    vpbroadcastq         m4, [srcq+r6   ]
    lea                srcq, [srcq+ssq*4]
    vpbroadcastq         m3, [srcq+ssq*0]
    vpbroadcastq         m5, [srcq+ssq*1]
    vpblendd             m1, m0, 0x30
    vpblendd             m0, m2, 0x30
    punpcklwd            m1, m0      ; 01 12
    vpbroadcastq         m0, [srcq+ssq*2]
    add                srcq, r6
    vpblendd             m2, m4, 0x30
    vpblendd             m4, m3, 0x30
    punpcklwd            m2, m4      ; 23 34
    vpblendd             m3, m5, 0x30
    vpblendd             m5, m0, 0x30
    punpcklwd            m3, m5      ; 45 56
.v_w4_loop:
    vpbroadcastq         m4, [srcq+ssq*0]
    pmaddwd              m5, m8, m1  ; a0 b0
    mova                 m1, m2
    pmaddwd              m2, m9      ; a1 b1
    paddd                m5, m6
    paddd                m5, m2
    mova                 m2, m3
    pmaddwd              m3, m10     ; a2 b2
    paddd                m5, m3
    vpblendd             m3, m0, m4, 0x30
    vpbroadcastq         m0, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    vpblendd             m4, m0, 0x30
    punpcklwd            m3, m4      ; 67 78
    pmaddwd              m4, m11, m3 ; a3 b3
    paddd                m5, m4
    psrad                m5, 6
    vextracti128        xm4, m5, 1
    packusdw            xm5, xm4
    pminsw              xm5, xm7
    movq       [dstq+dsq*0], xm5
    movhps     [dstq+dsq*1], xm5
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w4_loop
    RET
.v_w8:
    shl                  wd, 5
    mov                  r7, srcq
    mov                  r8, dstq
    lea                  wd, [hq+wq-256]
.v_w8_loop0:
    vbroadcasti128       m4, [srcq+ssq*0]
    vbroadcasti128       m5, [srcq+ssq*1]
    vbroadcasti128       m0, [srcq+r6   ]
    vbroadcasti128       m6, [srcq+ssq*2]
    lea                srcq, [srcq+ssq*4]
    vbroadcasti128       m1, [srcq+ssq*0]
    vbroadcasti128       m2, [srcq+ssq*1]
    vbroadcasti128       m3, [srcq+ssq*2]
    add                srcq, r6
    shufpd               m4, m0, 0x0c
    shufpd               m5, m1, 0x0c
    punpcklwd            m1, m4, m5 ; 01
    punpckhwd            m4, m5     ; 34
    shufpd               m6, m2, 0x0c
    punpcklwd            m2, m5, m6 ; 12
    punpckhwd            m5, m6     ; 45
    shufpd               m0, m3, 0x0c
    punpcklwd            m3, m6, m0 ; 23
    punpckhwd            m6, m0     ; 56
.v_w8_loop:
    vbroadcasti128      m14, [srcq+ssq*0]
    pmaddwd             m12, m8, m1  ; a0
    pmaddwd             m13, m8, m2  ; b0
    mova                 m1, m3
    mova                 m2, m4
    pmaddwd              m3, m9      ; a1
    pmaddwd              m4, m9      ; b1
    paddd               m12, m3
    paddd               m13, m4
    mova                 m3, m5
    mova                 m4, m6
    pmaddwd              m5, m10     ; a2
    pmaddwd              m6, m10     ; b2
    paddd               m12, m5
    vbroadcasti128       m5, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    paddd               m13, m6
    shufpd               m6, m0, m14, 0x0d
    shufpd               m0, m14, m5, 0x0c
    punpcklwd            m5, m6, m0  ; 67
    punpckhwd            m6, m0      ; 78
    pmaddwd             m14, m11, m5 ; a3
    paddd               m12, m14
    pmaddwd             m14, m11, m6 ; b3
    paddd               m13, m14
    psrad               m12, 5
    psrad               m13, 5
    packusdw            m12, m13
    pxor                m13, m13
    pavgw               m12, m13
    pminsw              m12, m7
    vpermq              m12, m12, q3120
    mova         [dstq+dsq*0], xm12
    vextracti128 [dstq+dsq*1], m12, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .v_w8_loop
    add                  r7, 16
    add                  r8, 16
    movzx                hd, wb
    mov                srcq, r7
    mov                dstq, r8
    sub                  wd, 1<<8
    jg .v_w8_loop0
    RET
.hv:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      16
    vpbroadcastw        m15, r8m
    cmp                  wd, 4
    jg .hv_w8
    movzx               mxd, mxb
    vpbroadcastd         m0, [base+subpel_filters+mxq*8+2]
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    vpbroadcastq         m1, [base+subpel_filters+myq*8]
    vpbroadcastd         m6, [pd_512]
    lea                  r6, [ssq*3]
    sub                srcq, 2
    sub                srcq, r6
    pxor                 m7, m7
    punpcklbw            m7, m0
    punpcklbw            m1, m1
    psraw                m1, 8 ; sign-extend
    test          dword r8m, 0x800
    jz .hv_10bit
    psraw                m7, 2
    psllw                m1, 2
.hv_10bit:
    pshufd              m11, m1, q0000
    pshufd              m12, m1, q1111
    pshufd              m13, m1, q2222
    pshufd              m14, m1, q3333
    cmp                  wd, 4
    je .hv_w4
    vbroadcasti128       m9, [subpel_h_shuf2]
    vbroadcasti128       m1, [srcq+r6   ]    ; 3 3
    movu                xm3, [srcq+ssq*2]
    movu                xm0, [srcq+ssq*0]
    movu                xm2, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*4]
    vinserti128          m3, [srcq+ssq*0], 1 ; 2 4
    vinserti128          m0, [srcq+ssq*1], 1 ; 0 5
    vinserti128          m2, [srcq+ssq*2], 1 ; 1 6
    add                srcq, r6
    pshufb               m1, m9
    pshufb               m3, m9
    pshufb               m0, m9
    pshufb               m2, m9
    pmaddwd              m1, m7
    pmaddwd              m3, m7
    pmaddwd              m0, m7
    pmaddwd              m2, m7
    phaddd               m1, m3
    phaddd               m0, m2
    paddd                m1, m6
    paddd                m0, m6
    psrad                m1, 10
    psrad                m0, 10
    packssdw             m1, m0         ; 3 2 0 1
    vextracti128        xm0, m1, 1      ; 3 4 5 6
    pshufd              xm2, xm1, q1301 ; 2 3 1 2
    pshufd              xm3, xm0, q2121 ; 4 5 4 5
    punpckhwd           xm1, xm2        ; 01 12
    punpcklwd           xm2, xm0        ; 23 34
    punpckhwd           xm3, xm0        ; 45 56
.hv_w2_loop:
    movu                xm4, [srcq+ssq*0]
    movu                xm5, [srcq+ssq*1]
    lea                srcq, [srcq+ssq*2]
    pshufb              xm4, xm9
    pshufb              xm5, xm9
    pmaddwd             xm4, xm7
    pmaddwd             xm5, xm7
    phaddd              xm4, xm5
    pmaddwd             xm5, xm11, xm1 ; a0 b0
    mova                xm1, xm2
    pmaddwd             xm2, xm12      ; a1 b1
    paddd               xm5, xm2
    mova                xm2, xm3
    pmaddwd             xm3, xm13      ; a2 b2
    paddd               xm5, xm3
    paddd               xm4, xm6
    psrad               xm4, 10
    packssdw            xm4, xm4
    palignr             xm3, xm4, xm0, 12
    mova                xm0, xm4
    punpcklwd           xm3, xm0       ; 67 78
    pmaddwd             xm4, xm14, xm3 ; a3 b3
    paddd               xm5, xm6
    paddd               xm5, xm4
    psrad               xm5, 10
    packusdw            xm5, xm5
    pminsw              xm5, xm15
    movd       [dstq+dsq*0], xm5
    pextrd     [dstq+dsq*1], xm5, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w2_loop
    RET
.hv_w4:
    vbroadcasti128       m9, [subpel_h_shufA]
    vbroadcasti128      m10, [subpel_h_shufB]
    pshufd               m8, m7, q1111
    pshufd               m7, m7, q0000
    movu                xm1, [srcq+ssq*0]
    vinserti128          m1, [srcq+ssq*1], 1     ; 0 1
    vbroadcasti128       m0, [srcq+r6   ]
    vinserti128          m2, m0, [srcq+ssq*2], 0 ; 2 3
    lea                srcq, [srcq+ssq*4]
    vinserti128          m0, [srcq+ssq*0], 1     ; 3 4
    movu                xm3, [srcq+ssq*1]
    vinserti128          m3, [srcq+ssq*2], 1     ; 5 6
    add                srcq, r6
    pshufb               m4, m1, m9
    pshufb               m1, m10
    pmaddwd              m4, m7
    pmaddwd              m1, m8
    pshufb               m5, m2, m9
    pshufb               m2, m10
    pmaddwd              m5, m7
    pmaddwd              m2, m8
    paddd                m4, m6
    paddd                m1, m4
    pshufb               m4, m0, m9
    pshufb               m0, m10
    pmaddwd              m4, m7
    pmaddwd              m0, m8
    paddd                m5, m6
    paddd                m2, m5
    pshufb               m5, m3, m9
    pshufb               m3, m10
    pmaddwd              m5, m7
    pmaddwd              m3, m8
    paddd                m4, m6
    paddd                m4, m0
    paddd                m5, m6
    paddd                m5, m3
    vperm2i128           m0, m1, m2, 0x21
    psrld                m1, 10
    psrld                m2, 10
    vperm2i128           m3, m4, m5, 0x21
    pslld                m4, 6
    pslld                m5, 6
    pblendw              m2, m4, 0xaa ; 23 34
    pslld                m0, 6
    pblendw              m1, m0, 0xaa ; 01 12
    psrld                m3, 10
    pblendw              m3, m5, 0xaa ; 45 56
    psrad                m0, m5, 16
.hv_w4_loop:
    movu                xm4, [srcq+ssq*0]
    vinserti128          m4, [srcq+ssq*1], 1
    lea                srcq, [srcq+ssq*2]
    pmaddwd              m5, m11, m1   ; a0 b0
    mova                 m1, m2
    pmaddwd              m2, m12       ; a1 b1
    paddd                m5, m6
    paddd                m5, m2
    mova                 m2, m3
    pmaddwd              m3, m13       ; a2 b2
    paddd                m5, m3
    pshufb               m3, m4, m9
    pshufb               m4, m10
    pmaddwd              m3, m7
    pmaddwd              m4, m8
    paddd                m3, m6
    paddd                m4, m3
    psrad                m4, 10
    packssdw             m0, m4        ; _ 7 6 8
    vpermq               m3, m0, q1122 ; _ 6 _ 7
    punpckhwd            m3, m0        ; 67 78
    mova                 m0, m4
    pmaddwd              m4, m14, m3   ; a3 b3
    paddd                m4, m5
    psrad                m4, 10
    vextracti128        xm5, m4, 1
    packusdw            xm4, xm5
    pminsw              xm4, xm15
    movq       [dstq+dsq*0], xm4
    movhps     [dstq+dsq*1], xm4
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w4_loop
    RET
.hv_w8:
    shr                 mxd, 16
    vpbroadcastq         m2, [base+subpel_filters+mxq*8]
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    pmovsxbw            xm1, [base+subpel_filters+myq*8]
    shl                  wd, 5
    lea                  r6, [ssq*3]
    sub                srcq, 6
    sub                srcq, r6
    pxor                 m0, m0
    punpcklbw            m0, m2
    mov                  r7, srcq
    mov                  r8, dstq
    lea                  wd, [hq+wq-256]
    test          dword r8m, 0x800
    jz .hv_w8_10bit
    psraw                m0, 2
    psllw               xm1, 2
.hv_w8_10bit:
    pshufd              m11, m0, q0000
    pshufd              m12, m0, q1111
    pshufd              m13, m0, q2222
    pshufd              m14, m0, q3333
%if WIN64
    %define v_mul (rsp+stack_offset+40) ; r4m
%else
    %define v_mul (rsp-24) ; red zone
%endif
    mova            [v_mul], xm1
.hv_w8_loop0:
%macro PUT_8TAP_HV_H 3 ; dst/src+0, src+8, src+16
    pshufb               m2, m%1, m9   ; 2 3 3 4 4 5 5 6
    pshufb              m%1, m8        ; 0 1 1 2 2 3 3 4
    pmaddwd              m3, m12, m2
    pmaddwd             m%1, m11
    pshufb              m%2, m9        ; 6 7 7 8 8 9 9 a
    shufpd               m2, m%2, 0x05 ; 4 5 5 6 6 7 7 8
    paddd                m3, m10
    paddd               m%1, m3
    pmaddwd              m3, m14, m%2
    paddd               m%1, m3
    pmaddwd              m3, m13, m2
    pshufb              m%3, m9        ; a b b c c d d e
    pmaddwd              m2, m11
    paddd               m%1, m3
    pmaddwd              m3, m12, m%2
    shufpd              m%2, m%3, 0x05 ; 8 9 9 a a b b c
    pmaddwd             m%3, m14
    pmaddwd             m%2, m13
    paddd                m2, m10
    paddd                m2, m3
    paddd               m%3, m2
    paddd               m%2, m%3
    psrad               m%1, 10
    psrad               m%2, 10
    packssdw            m%1, m%2
%endmacro
    movu                xm4, [srcq+r6 *1+ 0]
    vbroadcasti128       m8, [subpel_h_shufA]
    movu                xm6, [srcq+r6 *1+ 8]
    vbroadcasti128       m9, [subpel_h_shufB]
    movu                xm0, [srcq+r6 *1+16]
    vpbroadcastd        m10, [pd_512]
    movu                xm5, [srcq+ssq*0+ 0]
    vinserti128          m5, [srcq+ssq*4+ 0], 1
    movu                xm1, [srcq+ssq*0+16]
    vinserti128          m1, [srcq+ssq*4+16], 1
    shufpd               m7, m5, m1, 0x05
    INIT_XMM avx2
    PUT_8TAP_HV_H         4, 6, 0    ; 3
    INIT_YMM avx2
    PUT_8TAP_HV_H         5, 7, 1    ; 0 4
    movu                xm0, [srcq+ssq*2+ 0]
    vinserti128          m0, [srcq+r6 *2+ 0], 1
    movu                xm1, [srcq+ssq*2+16]
    vinserti128          m1, [srcq+r6 *2+16], 1
    shufpd               m7, m0, m1, 0x05
    PUT_8TAP_HV_H         0, 7, 1    ; 2 6
    movu                xm6, [srcq+ssq*1+ 0]
    movu                xm1, [srcq+ssq*1+16]
    lea                srcq, [srcq+ssq*4]
    vinserti128          m6, [srcq+ssq*1+ 0], 1
    vinserti128          m1, [srcq+ssq*1+16], 1
    add                srcq, r6
    shufpd               m7, m6, m1, 0x05
    PUT_8TAP_HV_H         6, 7, 1    ; 1 5
    vpermq               m4, m4, q1100
    vpermq               m5, m5, q3120
    vpermq               m6, m6, q3120
    vpermq               m7, m0, q3120
    punpcklwd            m3, m7, m4  ; 23
    punpckhwd            m4, m5      ; 34
    punpcklwd            m1, m5, m6  ; 01
    punpckhwd            m5, m6      ; 45
    punpcklwd            m2, m6, m7  ; 12
    punpckhwd            m6, m7      ; 56
.hv_w8_loop:
    vpbroadcastd         m9, [v_mul+4*0]
    vpbroadcastd         m7, [v_mul+4*1]
    vpbroadcastd        m10, [v_mul+4*2]
    pmaddwd              m8, m9, m1  ; a0
    pmaddwd              m9, m2      ; b0
    mova                 m1, m3
    mova                 m2, m4
    pmaddwd              m3, m7      ; a1
    pmaddwd              m4, m7      ; b1
    paddd                m8, m3
    paddd                m9, m4
    mova                 m3, m5
    mova                 m4, m6
    pmaddwd              m5, m10     ; a2
    pmaddwd              m6, m10     ; b2
    paddd                m8, m5
    paddd                m9, m6
    movu                xm5, [srcq+ssq*0]
    vinserti128          m5, [srcq+ssq*1], 1
    vbroadcasti128       m7, [subpel_h_shufA]
    vbroadcasti128      m10, [subpel_h_shufB]
    movu                xm6, [srcq+ssq*0+16]
    vinserti128          m6, [srcq+ssq*1+16], 1
    vextracti128     [dstq], m0, 1
    pshufb               m0, m5, m7  ; 01
    pshufb               m5, m10     ; 23
    pmaddwd              m0, m11
    pmaddwd              m5, m12
    paddd                m0, m5
    pshufb               m5, m6, m7  ; 89
    pshufb               m6, m10     ; ab
    pmaddwd              m5, m13
    pmaddwd              m6, m14
    paddd                m6, m5
    movu                xm5, [srcq+ssq*0+8]
    vinserti128          m5, [srcq+ssq*1+8], 1
    lea                srcq, [srcq+ssq*2]
    pshufb               m7, m5, m7
    pshufb               m5, m10
    pmaddwd             m10, m13, m7
    pmaddwd              m7, m11
    paddd                m0, m10
    vpbroadcastd        m10, [pd_512]
    paddd                m6, m7
    pmaddwd              m7, m14, m5
    pmaddwd              m5, m12
    paddd                m0, m7
    paddd                m5, m6
    vbroadcasti128       m6, [dstq]
    paddd                m8, m10
    paddd                m9, m10
    paddd                m0, m10
    paddd                m5, m10
    vpbroadcastd        m10, [v_mul+4*3]
    psrad                m0, 10
    psrad                m5, 10
    packssdw             m0, m5
    vpermq               m7, m0, q3120 ; 7 8
    shufpd               m6, m7, 0x04  ; 6 7
    punpcklwd            m5, m6, m7    ; 67
    punpckhwd            m6, m7        ; 78
    pmaddwd              m7, m10, m5   ; a3
    pmaddwd             m10, m6        ; b3
    paddd                m7, m8
    paddd                m9, m10
    psrad                m7, 10
    psrad                m9, 10
    packusdw             m7, m9
    pminsw               m7, m15
    vpermq               m7, m7, q3120
    mova         [dstq+dsq*0], xm7
    vextracti128 [dstq+dsq*1], m7, 1
    lea                dstq, [dstq+dsq*2]
    sub                  hd, 2
    jg .hv_w8_loop
    add                  r7, 16
    add                  r8, 16
    movzx                hd, wb
    mov                srcq, r7
    mov                dstq, r8
    sub                  wd, 1<<8
    jg .hv_w8_loop0
    RET

%if WIN64
DECLARE_REG_TMP 6, 4
%else
DECLARE_REG_TMP 6, 7
%endif

MC_8TAP_FN prep, sharp,          SHARP,   SHARP
MC_8TAP_FN prep, sharp_smooth,   SHARP,   SMOOTH
MC_8TAP_FN prep, smooth_sharp,   SMOOTH,  SHARP
MC_8TAP_FN prep, smooth,         SMOOTH,  SMOOTH
MC_8TAP_FN prep, sharp_regular,  SHARP,   REGULAR
MC_8TAP_FN prep, regular_sharp,  REGULAR, SHARP
MC_8TAP_FN prep, smooth_regular, SMOOTH,  REGULAR
MC_8TAP_FN prep, regular_smooth, REGULAR, SMOOTH
MC_8TAP_FN prep, regular,        REGULAR, REGULAR

cglobal prep_8tap_16bpc, 4, 8, 0, tmp, src, stride, w, h, mx, my
%define base r7-prep_avx2
    imul                mxd, mxm, 0x010101
    add                 mxd, t0d ; 8tap_h, mx, 4tap_h
    imul                myd, mym, 0x010101
    add                 myd, t1d ; 8tap_v, my, 4tap_v
    lea                  r7, [prep_avx2]
    movifnidn            hd, hm
    test                mxd, 0xf00
    jnz .h
    test                myd, 0xf00
    jnz .v
    tzcnt                wd, wd
    mov                 r6d, r7m ; bitdepth_max
    movzx                wd, word [r7+wq*2+table_offset(prep,)]
    vpbroadcastd         m5, [r7-prep_avx2+pw_8192]
    shr                 r6d, 11
    add                  wq, r7
    vpbroadcastd         m4, [base+prep_mul+r6*4]
    lea                  r6, [strideq*3]
%if WIN64
    pop                  r7
%endif
    jmp                  wq
.h_w4:
    movzx               mxd, mxb
    sub                srcq, 2
    pmovsxbw            xm0, [base+subpel_filters+mxq*8]
    vbroadcasti128       m3, [subpel_h_shufA]
    vbroadcasti128       m4, [subpel_h_shufB]
    WIN64_SPILL_XMM       8
    pshufd              xm0, xm0, q2211
    test          dword r7m, 0x800
    jnz .h_w4_12bpc
    psllw               xm0, 2
.h_w4_12bpc:
    vpbroadcastq         m6, xm0
    vpermq               m7, m0, q1111
.h_w4_loop:
    movu                xm1, [srcq+strideq*0]
    vinserti128          m1, [srcq+strideq*2], 1
    movu                xm2, [srcq+strideq*1]
    vinserti128          m2, [srcq+r6       ], 1
    lea                srcq, [srcq+strideq*4]
    pshufb               m0, m1, m3 ; 0 1 1 2 2 3 3 4
    pshufb               m1, m4     ; 2 3 3 4 4 5 5 6
    pmaddwd              m0, m6
    pmaddwd              m1, m7
    paddd                m0, m5
    paddd                m0, m1
    pshufb               m1, m2, m3
    pshufb               m2, m4
    pmaddwd              m1, m6
    pmaddwd              m2, m7
    paddd                m1, m5
    paddd                m1, m2
    psrad                m0, 4
    psrad                m1, 4
    packssdw             m0, m1
    mova             [tmpq], m0
    add                tmpq, 32
    sub                  hd, 4
    jg .h_w4_loop
    RET
.h:
    test                myd, 0xf00
    jnz .hv
    vpbroadcastd         m5, [prep_8tap_1d_rnd] ; 8 - (8192 << 4)
    lea                  r6, [strideq*3]
    cmp                  wd, 4
    je .h_w4
    shr                 mxd, 16
    sub                srcq, 6
    vpbroadcastq         m0, [base+subpel_filters+mxq*8]
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      12
    vbroadcasti128       m6, [subpel_h_shufA]
    vbroadcasti128       m7, [subpel_h_shufB]
    punpcklbw            m0, m0
    psraw                m0, 8 ; sign-extend
    test          dword r7m, 0x800
    jnz .h_12bpc
    psllw                m0, 2
.h_12bpc:
    pshufd               m8, m0, q0000
    pshufd               m9, m0, q1111
    pshufd              m10, m0, q2222
    pshufd              m11, m0, q3333
    cmp                  wd, 8
    jg .h_w16
.h_w8:
%macro PREP_8TAP_H 5 ; dst/src+0, src+8, src+16, tmp[1-2]
    pshufb              m%4, m%1, m7   ; 2 3 3 4 4 5 5 6
    pshufb              m%1, m6        ; 0 1 1 2 2 3 3 4
    pmaddwd             m%5, m9, m%4   ; abcd1
    pmaddwd             m%1, m8        ; abcd0
    pshufb              m%2, m7        ; 6 7 7 8 8 9 9 a
    shufpd              m%4, m%2, 0x05 ; 4 5 5 6 6 7 7 8
    paddd               m%5, m5
    paddd               m%1, m%5
    pmaddwd             m%5, m11, m%2  ; abcd3
    paddd               m%1, m%5
    pmaddwd             m%5, m10, m%4  ; abcd2
    pshufb              m%3, m7        ; a b b c c d d e
    pmaddwd             m%4, m8        ; efgh0
    paddd               m%1, m%5
    pmaddwd             m%5, m9, m%2   ; efgh1
    shufpd              m%2, m%3, 0x05 ; 8 9 9 a a b b c
    pmaddwd             m%3, m11       ; efgh3
    pmaddwd             m%2, m10       ; efgh2
    paddd               m%4, m5
    paddd               m%4, m%5
    paddd               m%3, m%4
    paddd               m%2, m%3
    psrad               m%1, 4
    psrad               m%2, 4
    packssdw            m%1, m%2
%endmacro
    movu                xm0, [srcq+strideq*0+ 0]
    vinserti128          m0, [srcq+strideq*1+ 0], 1
    movu                xm2, [srcq+strideq*0+16]
    vinserti128          m2, [srcq+strideq*1+16], 1
    lea                srcq, [srcq+strideq*2]
    shufpd               m1, m0, m2, 0x05
    PREP_8TAP_H           0, 1, 2, 3, 4
    mova             [tmpq], m0
    add                tmpq, 32
    sub                  hd, 2
    jg .h_w8
    RET
.h_w16:
    add                  wd, wd
.h_w16_loop0:
    mov                 r6d, wd
.h_w16_loop:
    movu                 m0, [srcq+r6-32]
    movu                 m1, [srcq+r6-24]
    movu                 m2, [srcq+r6-16]
    PREP_8TAP_H           0, 1, 2, 3, 4
    mova       [tmpq+r6-32], m0
    sub                 r6d, 32
    jg .h_w16_loop
    add                srcq, strideq
    add                tmpq, wq
    dec                  hd
    jg .h_w16_loop0
    RET
.v:
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    vpbroadcastq         m0, [base+subpel_filters+myq*8]
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      15
    vpbroadcastd         m7, [prep_8tap_1d_rnd]
    lea                  r6, [strideq*3]
    sub                srcq, r6
    punpcklbw            m0, m0
    psraw                m0, 8 ; sign-extend
    test          dword r7m, 0x800
    jnz .v_12bpc
    psllw                m0, 2
.v_12bpc:
    pshufd               m8, m0, q0000
    pshufd               m9, m0, q1111
    pshufd              m10, m0, q2222
    pshufd              m11, m0, q3333
    cmp                  wd, 4
    jg .v_w8
.v_w4:
    movq                xm1, [srcq+strideq*0]
    vpbroadcastq         m0, [srcq+strideq*1]
    vpbroadcastq         m2, [srcq+strideq*2]
    vpbroadcastq         m4, [srcq+r6       ]
    lea                srcq, [srcq+strideq*4]
    vpbroadcastq         m3, [srcq+strideq*0]
    vpbroadcastq         m5, [srcq+strideq*1]
    vpblendd             m1, m0, 0x30
    vpblendd             m0, m2, 0x30
    punpcklwd            m1, m0      ; 01 12
    vpbroadcastq         m0, [srcq+strideq*2]
    add                srcq, r6
    vpblendd             m2, m4, 0x30
    vpblendd             m4, m3, 0x30
    punpcklwd            m2, m4      ; 23 34
    vpblendd             m3, m5, 0x30
    vpblendd             m5, m0, 0x30
    punpcklwd            m3, m5      ; 45 56
.v_w4_loop:
    vpbroadcastq         m4, [srcq+strideq*0]
    pmaddwd              m5, m8, m1  ; a0 b0
    mova                 m1, m2
    pmaddwd              m2, m9      ; a1 b1
    paddd                m5, m7
    paddd                m5, m2
    mova                 m2, m3
    pmaddwd              m3, m10     ; a2 b2
    paddd                m5, m3
    vpblendd             m3, m0, m4, 0x30
    vpbroadcastq         m0, [srcq+strideq*1]
    lea                srcq, [srcq+strideq*2]
    vpblendd             m4, m0, 0x30
    punpcklwd            m3, m4      ; 67 78
    pmaddwd              m4, m11, m3 ; a3 b3
    paddd                m5, m4
    psrad                m5, 4
    vextracti128        xm4, m5, 1
    packssdw            xm5, xm4
    mova             [tmpq], xm5
    add                tmpq, 16
    sub                  hd, 2
    jg .v_w4_loop
    RET
.v_w8:
%if WIN64
    push                 r8
%endif
    mov                 r8d, wd
    shl                  wd, 5
    mov                  r5, srcq
    mov                  r7, tmpq
    lea                  wd, [hq+wq-256]
.v_w8_loop0:
    vbroadcasti128       m4, [srcq+strideq*0]
    vbroadcasti128       m5, [srcq+strideq*1]
    vbroadcasti128       m0, [srcq+r6       ]
    vbroadcasti128       m6, [srcq+strideq*2]
    lea                srcq, [srcq+strideq*4]
    vbroadcasti128       m1, [srcq+strideq*0]
    vbroadcasti128       m2, [srcq+strideq*1]
    vbroadcasti128       m3, [srcq+strideq*2]
    add                srcq, r6
    shufpd               m4, m0, 0x0c
    shufpd               m5, m1, 0x0c
    punpcklwd            m1, m4, m5 ; 01
    punpckhwd            m4, m5     ; 34
    shufpd               m6, m2, 0x0c
    punpcklwd            m2, m5, m6 ; 12
    punpckhwd            m5, m6     ; 45
    shufpd               m0, m3, 0x0c
    punpcklwd            m3, m6, m0 ; 23
    punpckhwd            m6, m0     ; 56
.v_w8_loop:
    vbroadcasti128      m14, [srcq+strideq*0]
    pmaddwd             m12, m8, m1  ; a0
    pmaddwd             m13, m8, m2  ; b0
    mova                 m1, m3
    mova                 m2, m4
    pmaddwd              m3, m9      ; a1
    pmaddwd              m4, m9      ; b1
    paddd               m12, m7
    paddd               m13, m7
    paddd               m12, m3
    paddd               m13, m4
    mova                 m3, m5
    mova                 m4, m6
    pmaddwd              m5, m10     ; a2
    pmaddwd              m6, m10     ; b2
    paddd               m12, m5
    vbroadcasti128       m5, [srcq+strideq*1]
    lea                srcq, [srcq+strideq*2]
    paddd               m13, m6
    shufpd               m6, m0, m14, 0x0d
    shufpd               m0, m14, m5, 0x0c
    punpcklwd            m5, m6, m0  ; 67
    punpckhwd            m6, m0      ; 78
    pmaddwd             m14, m11, m5 ; a3
    paddd               m12, m14
    pmaddwd             m14, m11, m6 ; b3
    paddd               m13, m14
    psrad               m12, 4
    psrad               m13, 4
    packssdw            m12, m13
    vpermq              m12, m12, q3120
    mova         [tmpq+r8*0], xm12
    vextracti128 [tmpq+r8*2], m12, 1
    lea                tmpq, [tmpq+r8*4]
    sub                  hd, 2
    jg .v_w8_loop
    add                  r5, 16
    add                  r7, 16
    movzx                hd, wb
    mov                srcq, r5
    mov                tmpq, r7
    sub                  wd, 1<<8
    jg .v_w8_loop0
%if WIN64
    pop                  r8
%endif
    RET
.hv:
    %assign stack_offset stack_offset - stack_size_padded
    WIN64_SPILL_XMM      16
    vpbroadcastd        m15, [prep_8tap_2d_rnd]
    cmp                  wd, 4
    jg .hv_w8
    movzx               mxd, mxb
    vpbroadcastd         m0, [base+subpel_filters+mxq*8+2]
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    vpbroadcastq         m1, [base+subpel_filters+myq*8]
    lea                  r6, [strideq*3]
    sub                srcq, 2
    sub                srcq, r6
    pxor                 m7, m7
    punpcklbw            m7, m0
    punpcklbw            m1, m1
    psraw                m7, 4
    psraw                m1, 8
    test          dword r7m, 0x800
    jz .hv_w4_10bit
    psraw                m7, 2
.hv_w4_10bit:
    pshufd              m11, m1, q0000
    pshufd              m12, m1, q1111
    pshufd              m13, m1, q2222
    pshufd              m14, m1, q3333
.hv_w4:
    vbroadcasti128       m9, [subpel_h_shufA]
    vbroadcasti128      m10, [subpel_h_shufB]
    pshufd               m8, m7, q1111
    pshufd               m7, m7, q0000
    movu                xm1, [srcq+strideq*0]
    vinserti128          m1, [srcq+strideq*1], 1     ; 0 1
    vbroadcasti128       m0, [srcq+r6       ]
    vinserti128          m2, m0, [srcq+strideq*2], 0 ; 2 3
    lea                srcq, [srcq+strideq*4]
    vinserti128          m0, [srcq+strideq*0], 1     ; 3 4
    movu                xm3, [srcq+strideq*1]
    vinserti128          m3, [srcq+strideq*2], 1     ; 5 6
    add                srcq, r6
    pshufb               m4, m1, m9
    pshufb               m1, m10
    pmaddwd              m4, m7
    pmaddwd              m1, m8
    pshufb               m5, m2, m9
    pshufb               m2, m10
    pmaddwd              m5, m7
    pmaddwd              m2, m8
    paddd                m4, m15
    paddd                m1, m4
    pshufb               m4, m0, m9
    pshufb               m0, m10
    pmaddwd              m4, m7
    pmaddwd              m0, m8
    paddd                m5, m15
    paddd                m2, m5
    pshufb               m5, m3, m9
    pshufb               m3, m10
    pmaddwd              m5, m7
    pmaddwd              m3, m8
    paddd                m4, m15
    paddd                m4, m0
    paddd                m5, m15
    paddd                m5, m3
    vperm2i128           m0, m1, m2, 0x21
    psrld                m1, 6
    psrld                m2, 6
    vperm2i128           m3, m4, m5, 0x21
    pslld                m4, 10
    pslld                m5, 10
    pblendw              m2, m4, 0xaa ; 23 34
    pslld                m0, 10
    pblendw              m1, m0, 0xaa ; 01 12
    psrld                m3, 6
    pblendw              m3, m5, 0xaa ; 45 56
    psrad                m0, m5, 16
.hv_w4_loop:
    movu                xm4, [srcq+strideq*0]
    vinserti128          m4, [srcq+strideq*1], 1
    lea                srcq, [srcq+strideq*2]
    pmaddwd              m5, m11, m1   ; a0 b0
    mova                 m1, m2
    pmaddwd              m2, m12       ; a1 b1
    paddd                m5, m15
    paddd                m5, m2
    mova                 m2, m3
    pmaddwd              m3, m13       ; a2 b2
    paddd                m5, m3
    pshufb               m3, m4, m9
    pshufb               m4, m10
    pmaddwd              m3, m7
    pmaddwd              m4, m8
    paddd                m3, m15
    paddd                m4, m3
    psrad                m4, 6
    packssdw             m0, m4        ; _ 7 6 8
    vpermq               m3, m0, q1122 ; _ 6 _ 7
    punpckhwd            m3, m0        ; 67 78
    mova                 m0, m4
    pmaddwd              m4, m14, m3   ; a3 b3
    paddd                m4, m5
    psrad                m4, 6
    vextracti128        xm5, m4, 1
    packssdw            xm4, xm5
    mova             [tmpq], xm4
    add                tmpq, 16
    sub                  hd, 2
    jg .hv_w4_loop
    RET
.hv_w8:
    shr                 mxd, 16
    vpbroadcastq         m2, [base+subpel_filters+mxq*8]
    movzx               mxd, myb
    shr                 myd, 16
    cmp                  hd, 4
    cmovle              myd, mxd
    pmovsxbw            xm1, [base+subpel_filters+myq*8]
%if WIN64
    PUSH                 r8
%endif
    mov                 r8d, wd
    shl                  wd, 5
    lea                  r6, [strideq*3]
    sub                srcq, 6
    sub                srcq, r6
    mov                  r5, srcq
    mov                  r7, tmpq
    lea                  wd, [hq+wq-256]
    pxor                 m0, m0
    punpcklbw            m0, m2
    mova            [v_mul], xm1
    psraw                m0, 4
    test          dword r7m, 0x800
    jz .hv_w8_10bit
    psraw                m0, 2
.hv_w8_10bit:
    pshufd              m11, m0, q0000
    pshufd              m12, m0, q1111
    pshufd              m13, m0, q2222
    pshufd              m14, m0, q3333
.hv_w8_loop0:
%macro PREP_8TAP_HV_H 3 ; dst/src+0, src+8, src+16
    pshufb               m2, m%1, m9   ; 2 3 3 4 4 5 5 6
    pshufb              m%1, m8        ; 0 1 1 2 2 3 3 4
    pmaddwd              m3, m12, m2
    pmaddwd             m%1, m11
    pshufb              m%2, m9        ; 6 7 7 8 8 9 9 a
    shufpd               m2, m%2, 0x05 ; 4 5 5 6 6 7 7 8
    paddd                m3, m15
    paddd               m%1, m3
    pmaddwd              m3, m14, m%2
    paddd               m%1, m3
    pmaddwd              m3, m13, m2
    pshufb              m%3, m9        ; a b b c c d d e
    pmaddwd              m2, m11
    paddd               m%1, m3
    pmaddwd              m3, m12, m%2
    shufpd              m%2, m%3, 0x05 ; 8 9 9 a a b b c
    pmaddwd             m%3, m14
    pmaddwd             m%2, m13
    paddd                m2, m15
    paddd                m2, m3
    paddd                m2, m%3
    paddd                m2, m%2
    psrad               m%1, 6
    psrad                m2, 6
    packssdw            m%1, m2
%endmacro
    movu                xm4, [srcq+r6       + 0]
    vbroadcasti128       m8, [subpel_h_shufA]
    movu                xm6, [srcq+r6       + 8]
    vbroadcasti128       m9, [subpel_h_shufB]
    movu                xm0, [srcq+r6       +16]
    movu                xm5, [srcq+strideq*0+ 0]
    vinserti128          m5, [srcq+strideq*4+ 0], 1
    movu                xm1, [srcq+strideq*0+16]
    vinserti128          m1, [srcq+strideq*4+16], 1
    shufpd               m7, m5, m1, 0x05
    INIT_XMM avx2
    PREP_8TAP_HV_H        4, 6, 0    ; 3
    INIT_YMM avx2
    PREP_8TAP_HV_H        5, 7, 1    ; 0 4
    movu                xm0, [srcq+strideq*2+ 0]
    vinserti128          m0, [srcq+r6     *2+ 0], 1
    movu                xm1, [srcq+strideq*2+16]
    vinserti128          m1, [srcq+r6     *2+16], 1
    shufpd               m7, m0, m1, 0x05
    PREP_8TAP_HV_H        0, 7, 1    ; 2 6
    movu                xm6, [srcq+strideq*1+ 0]
    movu                xm1, [srcq+strideq*1+16]
    lea                srcq, [srcq+strideq*4]
    vinserti128          m6, [srcq+strideq*1+ 0], 1
    vinserti128          m1, [srcq+strideq*1+16], 1
    add                srcq, r6
    shufpd               m7, m6, m1, 0x05
    PREP_8TAP_HV_H        6, 7, 1    ; 1 5
    vpermq               m4, m4, q1100
    vpermq               m5, m5, q3120
    vpermq               m6, m6, q3120
    vpermq               m7, m0, q3120
    punpcklwd            m3, m7, m4  ; 23
    punpckhwd            m4, m5      ; 34
    punpcklwd            m1, m5, m6  ; 01
    punpckhwd            m5, m6      ; 45
    punpcklwd            m2, m6, m7  ; 12
    punpckhwd            m6, m7      ; 56
.hv_w8_loop:
    vpbroadcastd         m9, [v_mul+4*0]
    vpbroadcastd         m7, [v_mul+4*1]
    vpbroadcastd        m10, [v_mul+4*2]
    pmaddwd              m8, m9, m1  ; a0
    pmaddwd              m9, m2      ; b0
    mova                 m1, m3
    mova                 m2, m4
    pmaddwd              m3, m7      ; a1
    pmaddwd              m4, m7      ; b1
    paddd                m8, m15
    paddd                m9, m15
    paddd                m8, m3
    paddd                m9, m4
    mova                 m3, m5
    mova                 m4, m6
    pmaddwd              m5, m10     ; a2
    pmaddwd              m6, m10     ; b2
    paddd                m8, m5
    paddd                m9, m6
    movu                xm5, [srcq+strideq*0]
    vinserti128          m5, [srcq+strideq*1], 1
    vbroadcasti128       m7, [subpel_h_shufA]
    vbroadcasti128      m10, [subpel_h_shufB]
    movu                xm6, [srcq+strideq*0+16]
    vinserti128          m6, [srcq+strideq*1+16], 1
    vextracti128     [tmpq], m0, 1
    pshufb               m0, m5, m7  ; 01
    pshufb               m5, m10     ; 23
    pmaddwd              m0, m11
    pmaddwd              m5, m12
    paddd                m0, m15
    paddd                m0, m5
    pshufb               m5, m6, m7  ; 89
    pshufb               m6, m10     ; ab
    pmaddwd              m5, m13
    pmaddwd              m6, m14
    paddd                m5, m15
    paddd                m6, m5
    movu                xm5, [srcq+strideq*0+8]
    vinserti128          m5, [srcq+strideq*1+8], 1
    lea                srcq, [srcq+strideq*2]
    pshufb               m7, m5, m7
    pshufb               m5, m10
    pmaddwd             m10, m13, m7
    pmaddwd              m7, m11
    paddd                m0, m10
    paddd                m6, m7
    pmaddwd              m7, m14, m5
    pmaddwd              m5, m12
    paddd                m0, m7
    paddd                m5, m6
    vbroadcasti128       m6, [tmpq]
    vpbroadcastd        m10, [v_mul+4*3]
    psrad                m0, 6
    psrad                m5, 6
    packssdw             m0, m5
    vpermq               m7, m0, q3120 ; 7 8
    shufpd               m6, m7, 0x04  ; 6 7
    punpcklwd            m5, m6, m7    ; 67
    punpckhwd            m6, m7        ; 78
    pmaddwd              m7, m10, m5   ; a3
    pmaddwd             m10, m6        ; b3
    paddd                m7, m8
    paddd                m9, m10
    psrad                m7, 6
    psrad                m9, 6
    packssdw             m7, m9
    vpermq               m7, m7, q3120
    mova         [tmpq+r8*0], xm7
    vextracti128 [tmpq+r8*2], m7, 1
    lea                tmpq, [tmpq+r8*4]
    sub                  hd, 2
    jg .hv_w8_loop
    add                  r5, 16
    add                  r7, 16
    movzx                hd, wb
    mov                srcq, r5
    mov                tmpq, r7
    sub                  wd, 1<<8
    jg .hv_w8_loop0
%if WIN64
    POP                  r8
%endif
    RET

%endif ; ARCH_X86_64
