; Copyright © 2022, VideoLAN and dav1d authors
; Copyright © 2022, Two Orioles, LLC
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

SECTION_RODATA 64

ipred_shuf:    db 14, 15, 14, 15,  0,  1,  2,  3,  6,  7,  6,  7,  0,  1,  2,  3
               db 10, 11, 10, 11,  8,  9, 10, 11,  2,  3,  2,  3,  8,  9, 10, 11
               db 12, 13, 12, 13,  4,  5,  6,  7,  4,  5,  4,  5,  4,  5,  6,  7
               db  8,  9,  8,  9, 12, 13, 14, 15,  0,  1,  0,  1, 12, 13, 14, 15
pw_1:          times 2 dw  1

%macro JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - 2*4)
    %xdefine %%base mangle(private_prefix %+ _%1_%2)
    %%table:
    %rep %0 - 2
        dd %%base %+ .%3 - (%%table - 2*4)
        %rotate 1
    %endrep
%endmacro

JMP_TABLE ipred_paeth_16bpc,      avx512icl, w4, w8, w16, w32, w64

SECTION .text

%macro PAETH 3 ; top, signed_ldiff, ldiff
    paddw               m0, m%2, m2
    psubw               m1, m0, m3  ; tldiff
    psubw               m0, m%1     ; tdiff
    pabsw               m1, m1
    pabsw               m0, m0
    pcmpgtw             k1, m0, m1
    pminsw              m0, m1
    pcmpgtw             k2, m%3, m0
    vpblendmw       m0{k1}, m%1, m3
    vpblendmw       m0{k2}, m2, m0
%endmacro

INIT_ZMM avx512icl
cglobal ipred_paeth_16bpc, 3, 7, 10, dst, stride, tl, w, h
%define base r6-ipred_paeth_16bpc_avx512icl_table
    lea                 r6, [ipred_paeth_16bpc_avx512icl_table]
    tzcnt               wd, wm
    movifnidn           hd, hm
    movsxd              wq, [r6+wq*4]
    vpbroadcastw        m3, [tlq]   ; topleft
    add                 wq, r6
    jmp                 wq
.w4:
    vpbroadcastq        m4, [tlq+2] ; top
    movsldup            m7, [base+ipred_shuf]
    lea                 r6, [strideq*3]
    psubw               m5, m4, m3
    pabsw               m6, m5
.w4_loop:
    sub                tlq, 16
    vbroadcasti32x4     m2, [tlq]
    pshufb              m2, m7      ; left
    PAETH                4, 5, 6
    vextracti32x4     xmm1, m0, 2
    vextracti32x4     xmm2, ym0, 1
    vextracti32x4     xmm3, m0, 3
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*1], xmm1
    movq   [dstq+strideq*2], xmm2
    movq   [dstq+r6       ], xmm3
    sub                 hd, 8
    jl .w4_end
    lea               dstq, [dstq+strideq*4]
    movhps [dstq+strideq*0], xm0
    movhps [dstq+strideq*1], xmm1
    movhps [dstq+strideq*2], xmm2
    movhps [dstq+r6       ], xmm3
    lea               dstq, [dstq+strideq*4]
    jg .w4_loop
.w4_end:
    RET
.w8:
    vbroadcasti32x4     m4, [tlq+2]
    movsldup            m7, [base+ipred_shuf]
    lea                 r6, [strideq*3]
    psubw               m5, m4, m3
    pabsw               m6, m5
.w8_loop:
    sub                tlq, 8
    vpbroadcastq        m2, [tlq]
    pshufb              m2, m7
    PAETH                4, 5, 6
    mova          [dstq+strideq*0], xm0
    vextracti32x4 [dstq+strideq*1], m0, 2
    vextracti32x4 [dstq+strideq*2], ym0, 1
    vextracti32x4 [dstq+r6       ], m0, 3
    lea               dstq, [dstq+strideq*4]
    sub                 hd, 4
    jg .w8_loop
    RET
.w16:
    vbroadcasti32x8     m4, [tlq+2]
    movsldup            m7, [base+ipred_shuf]
    psubw               m5, m4, m3
    pabsw               m6, m5
.w16_loop:
    sub                tlq, 4
    vpbroadcastd        m2, [tlq]
    pshufb              m2, m7
    PAETH                4, 5, 6
    mova          [dstq+strideq*0], ym0
    vextracti32x8 [dstq+strideq*1], m0, 1
    lea               dstq, [dstq+strideq*2]
    sub                 hd, 2
    jg .w16_loop
    RET
.w32:
    movu                m4, [tlq+2]
    psubw               m5, m4, m3
    pabsw               m6, m5
.w32_loop:
    sub                tlq, 2
    vpbroadcastw        m2, [tlq]
    PAETH                4, 5, 6
    mova            [dstq], m0
    add               dstq, strideq
    dec                 hd
    jg .w32_loop
    RET
.w64:
    movu                m4, [tlq+ 2]
    movu                m7, [tlq+66]
    psubw               m5, m4, m3
    psubw               m8, m7, m3
    pabsw               m6, m5
    pabsw               m9, m8
.w64_loop:
    sub                tlq, 2
    vpbroadcastw        m2, [tlq]
    PAETH                4, 5, 6
    mova       [dstq+64*0], m0
    PAETH                7, 8, 9
    mova       [dstq+64*1], m0
    add               dstq, strideq
    dec                 hd
    jg .w64_loop
    RET

%endif
