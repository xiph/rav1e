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

SECTION_RODATA

pw_2:             times 8 dw 2
pw_64:            times 8 dw 64
pw_256:           times 8 dw 256
pw_27615:         times 8 dw 27615
pd_65538:         times 2 dd 65538

bidir_rnd:        times 4 dw -16400
                  times 4 dw -16388
bidir_mul:        times 4 dw 2048
                  times 4 dw 8192

%macro BIDIR_JMP_TABLE 2-*
    %xdefine %1_%2_table (%%table - 2*%3)
    %xdefine %%base %1_%2_table
    %xdefine %%prefix mangle(private_prefix %+ _%1_16bpc_%2)
    %%table:
    %rep %0 - 2
        dd %%prefix %+ .w%3 - %%base
        %rotate 1
    %endrep
%endmacro

BIDIR_JMP_TABLE avg,        ssse3,    4, 8, 16, 32, 64, 128
BIDIR_JMP_TABLE w_avg,      ssse3,    4, 8, 16, 32, 64, 128
BIDIR_JMP_TABLE mask,       ssse3,    4, 8, 16, 32, 64, 128
BIDIR_JMP_TABLE w_mask_420, ssse3,    4, 8, 16, 32, 64, 128
BIDIR_JMP_TABLE w_mask_422, ssse3,    4, 8, 16, 32, 64, 128
BIDIR_JMP_TABLE w_mask_444, ssse3,    4, 8, 16, 32, 64, 128

SECTION .text

%if UNIX64
DECLARE_REG_TMP 7
%else
DECLARE_REG_TMP 5
%endif

INIT_XMM ssse3
%macro BIDIR_FN 0
    call .main
    jmp                  wq
.w4_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w4:
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    movq   [dstq+strideq*0], m1
    movhps [dstq+strideq*1], m1
    sub                  hd, 4
    jg .w4_loop
.ret:
    RET
.w8_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w8:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    sub                  hd, 2
    jne .w8_loop
    RET
.w16_loop:
    call .main
    add                dstq, strideq
.w16:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    dec                  hd
    jg .w16_loop
    RET
.w32_loop:
    call .main
    add                dstq, strideq
.w32:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    dec                  hd
    jg .w32_loop
    RET
.w64_loop:
    call .main
    add                dstq, strideq
.w64:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    call .main
    mova        [dstq+16*4], m0
    mova        [dstq+16*5], m1
    call .main
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m1
    dec                  hd
    jg .w64_loop
    RET
.w128_loop:
    call .main
    add                dstq, strideq
.w128:
    mova       [dstq+16* 0], m0
    mova       [dstq+16* 1], m1
    call .main
    mova       [dstq+16* 2], m0
    mova       [dstq+16* 3], m1
    call .main
    mova       [dstq+16* 4], m0
    mova       [dstq+16* 5], m1
    call .main
    mova       [dstq+16* 6], m0
    mova       [dstq+16* 7], m1
    call .main
    mova       [dstq+16* 8], m0
    mova       [dstq+16* 9], m1
    call .main
    mova       [dstq+16*10], m0
    mova       [dstq+16*11], m1
    call .main
    mova       [dstq+16*12], m0
    mova       [dstq+16*13], m1
    call .main
    mova       [dstq+16*14], m0
    mova       [dstq+16*15], m1
    dec                  hd
    jg .w128_loop
    RET
%endmacro

%if UNIX64
DECLARE_REG_TMP 7
%else
DECLARE_REG_TMP 5
%endif

cglobal avg_16bpc, 4, 7, 4, dst, stride, tmp1, tmp2, w, h
%define base r6-avg_ssse3_table
    LEA                  r6, avg_ssse3_table
    tzcnt                wd, wm
    mov                 t0d, r6m ; pixel_max
    movsxd               wq, [r6+wq*4]
    shr                 t0d, 11
    movddup              m2, [base+bidir_rnd+t0*8]
    movddup              m3, [base+bidir_mul+t0*8]
    movifnidn            hd, hm
    add                  wq, r6
    BIDIR_FN
ALIGN function_align
.main:
    mova                 m0, [tmp1q+16*0]
    paddsw               m0, [tmp2q+16*0]
    mova                 m1, [tmp1q+16*1]
    paddsw               m1, [tmp2q+16*1]
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    pmaxsw               m0, m2
    pmaxsw               m1, m2
    psubsw               m0, m2
    psubsw               m1, m2
    pmulhw               m0, m3
    pmulhw               m1, m3
    ret

cglobal w_avg_16bpc, 4, 7, 8, dst, stride, tmp1, tmp2, w, h
%define base r6-w_avg_ssse3_table
    LEA                  r6, w_avg_ssse3_table
    tzcnt                wd, wm
    mov                 t0d, r6m ; weight
    movd                 m6, r7m ; pixel_max
    movddup              m5, [base+pd_65538]
    movsxd               wq, [r6+wq*4]
    pshufb               m6, [base+pw_256]
    add                  wq, r6
    lea                 r6d, [t0-16]
    shl                 t0d, 16
    sub                 t0d, r6d ; 16-weight, weight
    paddw                m5, m6
    mov                 r6d, t0d
    shl                 t0d, 2
    test          dword r7m, 0x800
    cmovnz              r6d, t0d
    movifnidn            hd, hm
    movd                 m4, r6d
    pslld                m5, 7
    pxor                 m7, m7
    pshufd               m4, m4, q0000
    BIDIR_FN
ALIGN function_align
.main:
    mova                 m2, [tmp1q+16*0]
    mova                 m0, [tmp2q+16*0]
    punpckhwd            m3, m0, m2
    punpcklwd            m0, m2
    mova                 m2, [tmp1q+16*1]
    mova                 m1, [tmp2q+16*1]
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    pmaddwd              m3, m4
    pmaddwd              m0, m4
    paddd                m3, m5
    paddd                m0, m5
    psrad                m3, 8
    psrad                m0, 8
    packssdw             m0, m3
    punpckhwd            m3, m1, m2
    punpcklwd            m1, m2
    pmaddwd              m3, m4
    pmaddwd              m1, m4
    paddd                m3, m5
    paddd                m1, m5
    psrad                m3, 8
    psrad                m1, 8
    packssdw             m1, m3
    pminsw               m0, m6
    pminsw               m1, m6
    pmaxsw               m0, m7
    pmaxsw               m1, m7
    ret

%if ARCH_X86_64
cglobal mask_16bpc, 4, 7, 9, dst, stride, tmp1, tmp2, w, h, mask
%else
cglobal mask_16bpc, 4, 7, 8, dst, stride, tmp1, tmp2, w, mask
%define hd dword r5m
%define m8 [base+pw_64]
%endif
%define base r6-mask_ssse3_table
    LEA                  r6, mask_ssse3_table
    tzcnt                wd, wm
    mov                 t0d, r7m ; pixel_max
    shr                 t0d, 11
    movsxd               wq, [r6+wq*4]
    movddup              m6, [base+bidir_rnd+t0*8]
    movddup              m7, [base+bidir_mul+t0*8]
%if ARCH_X86_64
    mova                 m8, [base+pw_64]
    movifnidn            hd, hm
%endif
    add                  wq, r6
    mov               maskq, r6mp
    BIDIR_FN
ALIGN function_align
.main:
    movq                 m3, [maskq+8*0]
    mova                 m0, [tmp1q+16*0]
    mova                 m4, [tmp2q+16*0]
    pxor                 m5, m5
    punpcklbw            m3, m5
    punpckhwd            m2, m0, m4
    punpcklwd            m0, m4
    psubw                m1, m8, m3
    punpckhwd            m4, m3, m1 ; m, 64-m
    punpcklwd            m3, m1
    pmaddwd              m2, m4     ; tmp1 * m + tmp2 * (64-m)
    pmaddwd              m0, m3
    movq                 m3, [maskq+8*1]
    mova                 m1, [tmp1q+16*1]
    mova                 m4, [tmp2q+16*1]
    add               maskq, 8*2
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    psrad                m2, 5
    psrad                m0, 5
    packssdw             m0, m2
    punpcklbw            m3, m5
    punpckhwd            m2, m1, m4
    punpcklwd            m1, m4
    psubw                m5, m8, m3
    punpckhwd            m4, m3, m5 ; m, 64-m
    punpcklwd            m3, m5
    pmaddwd              m2, m4     ; tmp1 * m + tmp2 * (64-m)
    pmaddwd              m1, m3
    psrad                m2, 5
    psrad                m1, 5
    packssdw             m1, m2
    pmaxsw               m0, m6
    pmaxsw               m1, m6
    psubsw               m0, m6
    psubsw               m1, m6
    pmulhw               m0, m7
    pmulhw               m1, m7
    ret

cglobal w_mask_420_16bpc, 4, 7, 12, dst, stride, tmp1, tmp2, w, h, mask
%define base t0-w_mask_420_ssse3_table
    LEA                  t0, w_mask_420_ssse3_table
    tzcnt                wd, wm
    mov                 r6d, r8m ; pixel_max
    movd                 m0, r7m ; sign
    shr                 r6d, 11
    movsxd               wq, [t0+wq*4]
%if ARCH_X86_64
    mova                 m8, [base+pw_27615] ; ((64 - 38) << 10) + 1023 - 32
    mova                 m9, [base+pw_64]
    movddup             m10, [base+bidir_rnd+r6*8]
    movddup             m11, [base+bidir_mul+r6*8]
%else
    mova                 m1, [base+pw_27615] ; ((64 - 38) << 10) + 1023 - 32
    mova                 m2, [base+pw_64]
    movddup              m3, [base+bidir_rnd+r6*8]
    movddup              m4, [base+bidir_mul+r6*8]
    ALLOC_STACK       -16*4
    mova         [rsp+16*0], m1
    mova         [rsp+16*1], m2
    mova         [rsp+16*2], m3
    mova         [rsp+16*3], m4
    %define              m8  [rsp+gprsize+16*0]
    %define              m9  [rsp+gprsize+16*1]
    %define             m10  [rsp+gprsize+16*2]
    %define             m11  [rsp+gprsize+16*3]
%endif
    movd                 m7, [base+pw_2]
    psubw                m7, m0
    pshufb               m7, [base+pw_256]
    add                  wq, t0
    movifnidn            hd, r5m
    mov               maskq, r6mp
    call .main
    jmp                  wq
.w4_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 4
.w4:
    movq   [dstq+strideq*0], m0
    phaddw               m2, m3
    movhps [dstq+strideq*1], m0
    phaddd               m2, m2
    lea                dstq, [dstq+strideq*2]
    paddw                m2, m7
    movq   [dstq+strideq*0], m1
    psrlw                m2, 2
    movhps [dstq+strideq*1], m1
    packuswb             m2, m2
    movd            [maskq], m2
    sub                  hd, 4
    jg .w4_loop
    RET
.w8_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 4
.w8:
    mova   [dstq+strideq*0], m0
    paddw                m2, m3
    phaddw               m2, m2
    mova   [dstq+strideq*1], m1
    paddw                m2, m7
    psrlw                m2, 2
    packuswb             m2, m2
    movd            [maskq], m2
    sub                  hd, 2
    jg .w8_loop
    RET
.w16_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 8
.w16:
    mova [dstq+strideq*1+16*0], m2
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*1+16*1], m3
    mova [dstq+strideq*0+16*1], m1
    call .main
    paddw                m2, [dstq+strideq*1+16*0]
    paddw                m3, [dstq+strideq*1+16*1]
    mova [dstq+strideq*1+16*0], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*1], m1
    paddw                m2, m7
    psrlw                m2, 2
    packuswb             m2, m2
    movq            [maskq], m2
    sub                  hd, 2
    jg .w16_loop
    RET
.w32_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 16
.w32:
    mova [dstq+strideq*1+16*0], m2
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*1+16*1], m3
    mova [dstq+strideq*0+16*1], m1
    call .main
    mova [dstq+strideq*0+16*2], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*3], m2
    mova [dstq+strideq*0+16*3], m1
    call .main
    paddw                m2, [dstq+strideq*1+16*0]
    paddw                m3, [dstq+strideq*1+16*1]
    mova [dstq+strideq*1+16*0], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*2], m2
    mova [dstq+strideq*1+16*1], m1
    call .main
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16*2]
    paddw                m2, [dstq+strideq*1+16*3]
    mova [dstq+strideq*1+16*2], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16*3], m1
    packuswb             m3, m2
    mova            [maskq], m3
    sub                  hd, 2
    jg .w32_loop
    RET
.w64_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 16*2
.w64:
    mova [dstq+strideq*1+16*1], m2
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*1+16*2], m3
    mova [dstq+strideq*0+16*1], m1
    call .main
    mova [dstq+strideq*1+16*3], m2
    mova [dstq+strideq*0+16*2], m0
    mova [dstq+strideq*1+16*4], m3
    mova [dstq+strideq*0+16*3], m1
    call .main
    mova [dstq+strideq*1+16*5], m2
    mova [dstq+strideq*0+16*4], m0
    mova [dstq+strideq*1+16*6], m3
    mova [dstq+strideq*0+16*5], m1
    call .main
    mova [dstq+strideq*0+16*6], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*7], m2
    mova [dstq+strideq*0+16*7], m1
    call .main
    paddw                m2, [dstq+strideq*1+16*1]
    paddw                m3, [dstq+strideq*1+16*2]
    mova [dstq+strideq*1+16*0], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*2], m2
    mova [dstq+strideq*1+16*1], m1
    call .main
    paddw                m2, [dstq+strideq*1+16*3]
    paddw                m3, [dstq+strideq*1+16*4]
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16*2]
    mova [dstq+strideq*1+16*2], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16*3], m1
    packuswb             m3, m2
    mova       [maskq+16*0], m3
    call .main
    paddw                m2, [dstq+strideq*1+16*5]
    paddw                m3, [dstq+strideq*1+16*6]
    mova [dstq+strideq*1+16*4], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*6], m2
    mova [dstq+strideq*1+16*5], m1
    call .main
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16*6]
    paddw                m2, [dstq+strideq*1+16*7]
    mova [dstq+strideq*1+16*6], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16*7], m1
    packuswb             m3, m2
    mova       [maskq+16*1], m3
    sub                  hd, 2
    jg .w64_loop
    RET
.w128_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
    add               maskq, 16*4
.w128:
    mova [dstq+strideq*1+16* 1], m2
    mova [dstq+strideq*0+16* 0], m0
    mova [dstq+strideq*1+16* 2], m3
    mova [dstq+strideq*0+16* 1], m1
    call .main
    mova [dstq+strideq*1+16* 3], m2
    mova [dstq+strideq*0+16* 2], m0
    mova [dstq+strideq*1+16* 4], m3
    mova [dstq+strideq*0+16* 3], m1
    call .main
    mova [dstq+strideq*1+16* 5], m2
    mova [dstq+strideq*0+16* 4], m0
    mova [dstq+strideq*1+16* 6], m3
    mova [dstq+strideq*0+16* 5], m1
    call .main
    mova [dstq+strideq*1+16* 7], m2
    mova [dstq+strideq*0+16* 6], m0
    mova [dstq+strideq*1+16* 8], m3
    mova [dstq+strideq*0+16* 7], m1
    call .main
    mova [dstq+strideq*1+16* 9], m2
    mova [dstq+strideq*0+16* 8], m0
    mova [dstq+strideq*1+16*10], m3
    mova [dstq+strideq*0+16* 9], m1
    call .main
    mova [dstq+strideq*1+16*11], m2
    mova [dstq+strideq*0+16*10], m0
    mova [dstq+strideq*1+16*12], m3
    mova [dstq+strideq*0+16*11], m1
    call .main
    mova [dstq+strideq*1+16*13], m2
    mova [dstq+strideq*0+16*12], m0
    mova [dstq+strideq*1+16*14], m3
    mova [dstq+strideq*0+16*13], m1
    call .main
    mova [dstq+strideq*0+16*14], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*15], m2
    mova [dstq+strideq*0+16*15], m1
    call .main
    paddw                m2, [dstq+strideq*1+16* 1]
    paddw                m3, [dstq+strideq*1+16* 2]
    mova [dstq+strideq*1+16* 0], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16* 2], m2
    mova [dstq+strideq*1+16* 1], m1
    call .main
    paddw                m2, [dstq+strideq*1+16* 3]
    paddw                m3, [dstq+strideq*1+16* 4]
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16* 2]
    mova [dstq+strideq*1+16* 2], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16* 3], m1
    packuswb             m3, m2
    mova       [maskq+16*0], m3
    call .main
    paddw                m2, [dstq+strideq*1+16* 5]
    paddw                m3, [dstq+strideq*1+16* 6]
    mova [dstq+strideq*1+16* 4], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16* 6], m2
    mova [dstq+strideq*1+16* 5], m1
    call .main
    paddw                m2, [dstq+strideq*1+16* 7]
    paddw                m3, [dstq+strideq*1+16* 8]
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16* 6]
    mova [dstq+strideq*1+16* 6], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16* 7], m1
    packuswb             m3, m2
    mova       [maskq+16*1], m3
    call .main
    paddw                m2, [dstq+strideq*1+16* 9]
    paddw                m3, [dstq+strideq*1+16*10]
    mova [dstq+strideq*1+16* 8], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*10], m2
    mova [dstq+strideq*1+16* 9], m1
    call .main
    paddw                m2, [dstq+strideq*1+16*11]
    paddw                m3, [dstq+strideq*1+16*12]
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16*10]
    mova [dstq+strideq*1+16*10], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16*11], m1
    packuswb             m3, m2
    mova       [maskq+16*2], m3
    call .main
    paddw                m2, [dstq+strideq*1+16*13]
    paddw                m3, [dstq+strideq*1+16*14]
    mova [dstq+strideq*1+16*12], m0
    phaddw               m2, m3
    mova [dstq+strideq*1+16*14], m2
    mova [dstq+strideq*1+16*13], m1
    call .main
    phaddw               m2, m3
    paddw                m3, m7, [dstq+strideq*1+16*14]
    paddw                m2, [dstq+strideq*1+16*15]
    mova [dstq+strideq*1+16*14], m0
    paddw                m2, m7
    psrlw                m3, 2
    psrlw                m2, 2
    mova [dstq+strideq*1+16*15], m1
    packuswb             m3, m2
    mova       [maskq+16*3], m3
    sub                  hd, 2
    jg .w128_loop
    RET
ALIGN function_align
.main:
%macro W_MASK 2 ; dst/tmp_offset, mask
    mova                m%1, [tmp1q+16*%1]
    mova                m%2, [tmp2q+16*%1]
    punpcklwd            m4, m%2, m%1
    punpckhwd            m5, m%2, m%1
    psubsw              m%1, m%2
    pabsw               m%1, m%1
    psubusw              m6, m8, m%1
    psrlw                m6, 10      ; 64-m
    psubw               m%2, m9, m6  ; m
    punpcklwd           m%1, m6, m%2
    punpckhwd            m6, m%2
    pmaddwd             m%1, m4
    pmaddwd              m6, m5
    psrad               m%1, 5
    psrad                m6, 5
    packssdw            m%1, m6
    pmaxsw              m%1, m10
    psubsw              m%1, m10
    pmulhw              m%1, m11
%endmacro
    W_MASK                0, 2
    W_MASK                1, 3
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    ret

cglobal w_mask_422_16bpc, 4, 7, 12, dst, stride, tmp1, tmp2, w, h, mask
%define base t0-w_mask_422_ssse3_table
    LEA                  t0, w_mask_422_ssse3_table
    tzcnt                wd, wm
    mov                 r6d, r8m ; pixel_max
    movd                 m7, r7m ; sign
    shr                 r6d, 11
    movsxd               wq, [t0+wq*4]
%if ARCH_X86_64
    mova                 m8, [base+pw_27615]
    mova                 m9, [base+pw_64]
    movddup             m10, [base+bidir_rnd+r6*8]
    movddup             m11, [base+bidir_mul+r6*8]
%else
    mova                 m1, [base+pw_27615]
    mova                 m2, [base+pw_64]
    movddup              m3, [base+bidir_rnd+r6*8]
    movddup              m4, [base+bidir_mul+r6*8]
    ALLOC_STACK       -16*4
    mova         [rsp+16*0], m1
    mova         [rsp+16*1], m2
    mova         [rsp+16*2], m3
    mova         [rsp+16*3], m4
%endif
    pxor                 m0, m0
    add                  wq, t0
    pshufb               m7, m0
    movifnidn            hd, r5m
    mov               maskq, r6mp
    call .main
    jmp                  wq
.w4_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w4:
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    movq   [dstq+strideq*0], m1
    movhps [dstq+strideq*1], m1
    sub                  hd, 4
    jg .w4_loop
.end:
    RET
.w8_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w8:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    sub                  hd, 2
    jg .w8_loop
.w8_end:
    RET
.w16_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w16:
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m1
    call .main
    mova [dstq+strideq*1+16*0], m0
    mova [dstq+strideq*1+16*1], m1
    sub                  hd, 2
    jg .w16_loop
    RET
.w32_loop:
    call .main
    add                dstq, strideq
.w32:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    dec                  hd
    jg .w32_loop
    RET
.w64_loop:
    call .main
    add                dstq, strideq
.w64:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    call .main
    mova        [dstq+16*4], m0
    mova        [dstq+16*5], m1
    call .main
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m1
    dec                  hd
    jg .w64_loop
    RET
.w128_loop:
    call .main
    add                dstq, strideq
.w128:
    mova       [dstq+16* 0], m0
    mova       [dstq+16* 1], m1
    call .main
    mova       [dstq+16* 2], m0
    mova       [dstq+16* 3], m1
    call .main
    mova       [dstq+16* 4], m0
    mova       [dstq+16* 5], m1
    call .main
    mova       [dstq+16* 6], m0
    mova       [dstq+16* 7], m1
    call .main
    mova       [dstq+16* 8], m0
    mova       [dstq+16* 9], m1
    call .main
    mova       [dstq+16*10], m0
    mova       [dstq+16*11], m1
    call .main
    mova       [dstq+16*12], m0
    mova       [dstq+16*13], m1
    call .main
    mova       [dstq+16*14], m0
    mova       [dstq+16*15], m1
    dec                  hd
    jg .w128_loop
    RET
ALIGN function_align
.main:
    W_MASK                0, 2
    W_MASK                1, 3
    phaddw               m2, m3
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    packuswb             m2, m2
    pxor                 m3, m3
    psubb                m2, m7
    pavgb                m2, m3
    movq            [maskq], m2
    add               maskq, 8
    ret

cglobal w_mask_444_16bpc, 4, 7, 12, dst, stride, tmp1, tmp2, w, h, mask
%define base t0-w_mask_444_ssse3_table
    LEA                  t0, w_mask_444_ssse3_table
    tzcnt                wd, wm
    mov                 r6d, r8m ; pixel_max
    shr                 r6d, 11
    movsxd               wq, [t0+wq*4]
%if ARCH_X86_64
    mova                 m8, [base+pw_27615]
    mova                 m9, [base+pw_64]
    movddup             m10, [base+bidir_rnd+r6*8]
    movddup             m11, [base+bidir_mul+r6*8]
%else
    mova                 m1, [base+pw_27615]
    mova                 m2, [base+pw_64]
    movddup              m3, [base+bidir_rnd+r6*8]
    movddup              m7, [base+bidir_mul+r6*8]
    ALLOC_STACK       -16*3
    mova         [rsp+16*0], m1
    mova         [rsp+16*1], m2
    mova         [rsp+16*2], m3
    %define             m11  m7
%endif
    add                  wq, t0
    movifnidn            hd, r5m
    mov               maskq, r6mp
    call .main
    jmp                  wq
.w4_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w4:
    movq   [dstq+strideq*0], m0
    movhps [dstq+strideq*1], m0
    lea                dstq, [dstq+strideq*2]
    movq   [dstq+strideq*0], m1
    movhps [dstq+strideq*1], m1
    sub                  hd, 4
    jg .w4_loop
.end:
    RET
.w8_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w8:
    mova   [dstq+strideq*0], m0
    mova   [dstq+strideq*1], m1
    sub                  hd, 2
    jg .w8_loop
.w8_end:
    RET
.w16_loop:
    call .main
    lea                dstq, [dstq+strideq*2]
.w16:
    mova [dstq+strideq*0+16*0], m0
    mova [dstq+strideq*0+16*1], m1
    call .main
    mova [dstq+strideq*1+16*0], m0
    mova [dstq+strideq*1+16*1], m1
    sub                  hd, 2
    jg .w16_loop
    RET
.w32_loop:
    call .main
    add                dstq, strideq
.w32:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    dec                  hd
    jg .w32_loop
    RET
.w64_loop:
    call .main
    add                dstq, strideq
.w64:
    mova        [dstq+16*0], m0
    mova        [dstq+16*1], m1
    call .main
    mova        [dstq+16*2], m0
    mova        [dstq+16*3], m1
    call .main
    mova        [dstq+16*4], m0
    mova        [dstq+16*5], m1
    call .main
    mova        [dstq+16*6], m0
    mova        [dstq+16*7], m1
    dec                  hd
    jg .w64_loop
    RET
.w128_loop:
    call .main
    add                dstq, strideq
.w128:
    mova       [dstq+16* 0], m0
    mova       [dstq+16* 1], m1
    call .main
    mova       [dstq+16* 2], m0
    mova       [dstq+16* 3], m1
    call .main
    mova       [dstq+16* 4], m0
    mova       [dstq+16* 5], m1
    call .main
    mova       [dstq+16* 6], m0
    mova       [dstq+16* 7], m1
    call .main
    mova       [dstq+16* 8], m0
    mova       [dstq+16* 9], m1
    call .main
    mova       [dstq+16*10], m0
    mova       [dstq+16*11], m1
    call .main
    mova       [dstq+16*12], m0
    mova       [dstq+16*13], m1
    call .main
    mova       [dstq+16*14], m0
    mova       [dstq+16*15], m1
    dec                  hd
    jg .w128_loop
    RET
ALIGN function_align
.main:
    W_MASK                0, 2
    W_MASK                1, 3
    packuswb             m2, m3
    add               tmp1q, 16*2
    add               tmp2q, 16*2
    mova            [maskq], m2
    add               maskq, 16
    ret
