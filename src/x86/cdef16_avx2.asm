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

dir_shift: times 2 dw 0x4000
           times 2 dw 0x1000

cextern cdef_dir_8bpc_avx2.main

SECTION .text

%macro REPX 2-*
    %xdefine %%f(x) %1
%rep %0 - 1
    %rotate 1
    %%f(%1)
%endrep
%endmacro

INIT_YMM avx2
cglobal cdef_dir_16bpc, 4, 7, 6, src, stride, var, bdmax
    lea             r6, [dir_shift]
    shr         bdmaxd, 11 ; 0 for 10bpc, 1 for 12bpc
    vpbroadcastd    m4, [r6+bdmaxq*4]
    lea             r6, [strideq*3]
    mova           xm0, [srcq+strideq*0]
    mova           xm1, [srcq+strideq*1]
    mova           xm2, [srcq+strideq*2]
    mova           xm3, [srcq+r6       ]
    lea           srcq, [srcq+strideq*4]
    vinserti128     m0, [srcq+r6       ], 1
    vinserti128     m1, [srcq+strideq*2], 1
    vinserti128     m2, [srcq+strideq*1], 1
    vinserti128     m3, [srcq+strideq*0], 1
    REPX {pmulhuw x, m4}, m0, m1, m2, m3
    jmp mangle(private_prefix %+ _cdef_dir_8bpc %+ SUFFIX).main

%endif ; ARCH_X86_64
