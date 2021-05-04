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

put_bilin_h_rnd:  dw  8,  8, 10, 10

pw_2:     times 2 dw 2
pw_16:    times 2 dw 16
pw_2048:  times 2 dw 2048
pw_8192:  times 2 dw 8192

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

BASE_JMP_TABLE put,  avx2, 2, 4, 8, 16, 32, 64, 128

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

%define table_offset(type, fn) type %+ fn %+ SUFFIX %+ _table - type %+ SUFFIX

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

%endif ; ARCH_X86_64
