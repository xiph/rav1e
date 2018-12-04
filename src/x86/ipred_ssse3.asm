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

%macro JMP_TABLE 3-*
    %xdefine %1_%2_table (%%table - 2*4)
    %xdefine %%base mangle(private_prefix %+ _%1_%2)
    %%table:
    %rep %0 - 2
        dd %%base %+ .%3 - (%%table - 2*4)
        %rotate 1
    %endrep
%endmacro

JMP_TABLE      ipred_h,  ssse3, w4, w8, w16, w32, w64

SECTION .text


%macro IPRED_SET   4                                          ; width, store_type, stride, stride size, pshuflw_imm8
    pshuflw                      m1, m0, %4                   ; extend 8 byte for 2 pos
    punpcklqdq                   m1, m1
    mov%2          [dstq +      %3], m1
%if %1 > 16
    mov%2          [dstq + 16 + %3], m1
%endif
%if %1 > 32
    mov%2          [dstq + 32 + %3], m1
    mov%2          [dstq + 48 + %3], m1
%endif
%endmacro

%macro IPRED_H   3                                          ; width, loop label, store_type
    sub                         tlq, 4
    movd                         m0, [tlq]                  ; get 4 bytes of topleft data
    punpcklbw                    m0, m0                     ; extend 2 byte
    IPRED_SET                    %1, %3,         0, q3333
    IPRED_SET                    %1, %3,   strideq, q2222
    IPRED_SET                    %1, %3, strideq*2, q1111
    IPRED_SET                    %1, %3,  stride3q, q0000
    lea                        dstq, [dstq+strideq*4]
    sub                          hd, 4
    jg   %2
    RET
%endmacro

INIT_XMM ssse3
cglobal ipred_h, 3, 6, 2, dst, stride, tl, w, h, stride3
    lea                          r5, [ipred_h_ssse3_table]
    tzcnt                        wd, wm
    movifnidn                    hd, hm
%if ARCH_X86_64
    movsxd                       wq, [r5+wq*4]
%else
    mov                          wq, [r5+wq*4]
%endif
    add                          wq, r5
    lea                    stride3q, [strideq*3]
    jmp                          wq
.w4:
    IPRED_H                       4,  .w4, d
.w8:
    IPRED_H                       8,  .w8, q
.w16:
    IPRED_H                      16, .w16, u
.w32:
    IPRED_H                      32, .w32, u
.w64:
    IPRED_H                      64, .w64, u
