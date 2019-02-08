; Copyright © 2018, VideoLAN and dav1d authors
; Copyright © 2018, Two Orioles, LLC
; Copyright © 2019, VideoLabs
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

pb_0: times 16 db 0
pw_256: times 8 dw 256
pw_2048: times 8 dw 2048
pw_0x7FFF: times 8 dw 0x7FFF
tap_table: dw 4, 2, 3, 3, 2, 1
           db -1 * 16 + 1, -2 * 16 + 2
           db  0 * 16 + 1, -1 * 16 + 2
           db  0 * 16 + 1,  0 * 16 + 2
           db  0 * 16 + 1,  1 * 16 + 2
           db  1 * 16 + 1,  2 * 16 + 2
           db  1 * 16 + 0,  2 * 16 + 1
           db  1 * 16 + 0,  2 * 16 + 0
           db  1 * 16 + 0,  2 * 16 - 1
           ; the last 6 are repeats of the first 6 so we don't need to & 7
           db -1 * 16 + 1, -2 * 16 + 2
           db  0 * 16 + 1, -1 * 16 + 2
           db  0 * 16 + 1,  0 * 16 + 2
           db  0 * 16 + 1,  1 * 16 + 2
           db  1 * 16 + 1,  2 * 16 + 2
           db  1 * 16 + 0,  2 * 16 + 1

SECTION .text

INIT_XMM ssse3

%macro movif32 2
 %if ARCH_X86_32
    mov     %1, %2
 %endif
%endmacro

%macro SAVE_ARG 2   ; varname, argnum
 %define %1_stkloc  [rsp+%2*gprsize]
 %define %1_argnum  %2
    mov             r2, r%2m
    mov      %1_stkloc, r2
%endmacro

%macro LOAD_ARG 1-2 0 ; varname, load_to_varname_register
 %if %2 == 0
    mov r %+ %{1}_argnum, %1_stkloc
 %else
    mov            %1q, %1_stkloc
 %endif
%endmacro

%macro LOAD_ARG32 1-2 ; varname, load_to_varname_register
 %if ARCH_X86_32
  %if %0 == 1
    LOAD_ARG %1
  %else
    LOAD_ARG %1, %2
  %endif
 %endif
%endmacro

%if ARCH_X86_32
 %define PIC_base_offset $$
 %define PIC_sym(sym) (PIC_reg+(sym)-PIC_base_offset)
%else
 %define PIC_sym(sym) sym
%endif

%macro SAVE_PIC_REG 1
 %if ARCH_X86_32
    mov       [esp+%1], PIC_reg
 %endif
%endmacro

%macro LOAD_PIC_REG 1
 %if ARCH_X86_32
    mov        PIC_reg, [esp+%1]
 %endif
%endmacro

%macro ACCUMULATE_TAP 6 ; tap_offset, shift, strength, mul_tap, w, stride
 %if ARCH_X86_64
    ; load p0/p1
    movsx         offq, byte [dirq+kq+%1]       ; off1
  %if %5 == 4
    movq            m5, [stkq+offq*2+%6*0]      ; p0
    movhps          m5, [stkq+offq*2+%6*1]
  %else
    movu            m5, [stkq+offq*2+%6*0]      ; p0
  %endif
    neg           offq                          ; -off1
  %if %5 == 4
    movq            m6, [stkq+offq*2+%6*0]      ; p1
    movhps          m6, [stkq+offq*2+%6*1]
  %else
    movu            m6, [stkq+offq*2+%6*0]      ; p1
  %endif
    pcmpeqw         m9, m14, m5
    pcmpeqw        m10, m14, m6
    pandn           m9, m5
    pandn          m10, m6
    pmaxsw          m7, m9                      ; max after p0
    pminsw          m8, m5                      ; min after p0
    pmaxsw          m7, m10                     ; max after p1
    pminsw          m8, m6                      ; min after p1

    ; accumulate sum[m13] over p0/p1
    psubw           m5, m4                      ; diff_p0(p0 - px)
    psubw           m6, m4                      ; diff_p1(p1 - px)
    pabsw           m9, m5
    pabsw          m10, m6
    mova           m12, m9
    psrlw           m9, %2
    psignw         m11, %4, m5
    psubusw         m5, %3, m9
    mova            m9, m10
    pminsw          m5, m12                     ; constrain(diff_p0)
    psrlw          m10, %2
    psignw         m12, %4, m6
    psubusw         m6, %3, m10
    pmullw          m5, m11                     ; constrain(diff_p0) * taps
    pminsw          m6, m9                      ; constrain(diff_p1)
    pmullw          m6, m12                     ; constrain(diff_p1) * taps
    paddw          m13, m5
    paddw          m13, m6
 %else
    ; load p0
    movsx         offq, byte [dirq+kq+%1]       ; off1
  %if %5 == 4
    movq            m5, [stkq+offq*2+%6*0]      ; p0
    movhps          m5, [stkq+offq*2+%6*1]
  %else
    movu            m5, [stkq+offq*2+%6*0]      ; p0
  %endif
    pcmpeqw         m3, m5, [PIC_sym(pw_0x7FFF)]
    pandn           m3, m5
    pmaxsw          m7, m3                      ; max after p0
    pminsw          m8, m5                      ; min after p0

    ; accumulate sum[m7] over p0
    psubw           m5, m4                      ; diff_p0(p0 - px)
    psignw          m6, %4, m5                      ; constrain(diff_p0)
    pabsw           m5, m5
    mova            m3, m5
    psrlw           m5, %2
    paddsw          m5, %3
    pandn           m5, [PIC_sym(pw_0x7FFF)]
    pminsw          m5, m3
    pmullw          m5, m6                      ; constrain(diff_p0) * taps
    paddw          m13, m5

    ; load p1
    neg           offq                          ; -off1
  %if %5 == 4
    movq            m5, [stkq+offq*2+%6*0]      ; p1
    movhps          m5, [stkq+offq*2+%6*1]
  %else
    movu            m5, [stkq+offq*2+%6*0]      ; p1
  %endif
    pcmpeqw         m3, m5, [PIC_sym(pw_0x7FFF)]
    pandn           m3, m5
    pmaxsw          m7, m3                      ; max after p1
    pminsw          m8, m5                      ; min after p1

    ; accumulate sum[m7] over p1
    psubw           m5, m4                      ; diff_p1(p1 - px)
    psignw          m6, %4, m5                  ; constrain(diff_p1)
    pabsw           m5, m5
    mova            m3, m5
    psrlw           m5, %2
    paddsw          m5, %3
    pandn           m5, [PIC_sym(pw_0x7FFF)]
    pminsw          m5, m3
    pmullw          m5, m6                      ; constrain(diff_p1) * taps
    paddw          m13, m5
 %endif
%endmacro

%macro PMOVZXBW 2-3 0 ; %3 = half
 %if %3 == 1
    movd            %1, %2
 %else
    movq            %1, %2
 %endif
    punpcklbw       %1, m15
%endmacro

%macro LOAD_BODY 4  ; dst, src, block_width, tmp_stride
 %if %3 == 4
    PMOVZXBW        m0, [%2+strideq*0]
    PMOVZXBW        m1, [%2+strideq*1]
    PMOVZXBW        m2, [%2+strideq*2]
    PMOVZXBW        m3, [%2+stride3q]
 %else
    movu            m0, [%2+strideq*0]
    movu            m1, [%2+strideq*1]
    movu            m2, [%2+strideq*2]
    movu            m3, [%2+stride3q]
    punpckhbw       m4, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m5, m1, m15
    punpcklbw       m1, m15
    punpckhbw       m6, m2, m15
    punpcklbw       m2, m15
    punpckhbw       m7, m3, m15
    punpcklbw       m3, m15
 %endif
    mova     [%1+0*%4], m0
    mova     [%1+1*%4], m1
    mova     [%1+2*%4], m2
    mova     [%1+3*%4], m3
 %if %3 == 8
    mova [%1+0*%4+2*8], m4
    mova [%1+1*%4+2*8], m5
    mova [%1+2*%4+2*8], m6
    mova [%1+3*%4+2*8], m7
 %endif
%endmacro

%macro cdef_filter_fn 3 ; w, h, stride
 %if ARCH_X86_64
cglobal cdef_filter_%1x%2, 4, 9, 16, 3 * 16 + (%2+4)*%3, \
                           dst, stride, left, top, pri, sec, stride3, dst4, edge
    pcmpeqw        m14, m14
    psrlw          m14, 1                   ; 0x7FFF
    pxor           m15, m15

  %define px rsp+3*16+2*%3
 %else
cglobal cdef_filter_%1x%2, 2, 7, 8, - 5 * 16 - (%2+4)*%3, \
                           dst, stride, left, top, stride3, dst4, edge
    SAVE_ARG      left, 2
    SAVE_ARG       top, 3
    SAVE_ARG       pri, 4
    SAVE_ARG       sec, 5
    SAVE_ARG       dir, 6
    SAVE_ARG   damping, 7

  %define PIC_reg r2
    LEA        PIC_reg, PIC_base_offset

  %define m15 [PIC_sym(pb_0)]

  %define px esp+5*16+2*%3
 %endif

    mov          edged, r8m

    ; prepare pixel buffers - body/right
 %if %2 == 8
    lea          dst4q, [dstq+strideq*4]
 %endif
    lea       stride3q, [strideq*3]
    test         edged, 2                   ; have_right
    jz .no_right
    LOAD_BODY       px, dstq, %1, %3
 %if %2 == 8
    LOAD_BODY  px+4*%3, dst4q, %1, %3
 %endif
    jmp .body_done
.no_right:
    PMOVZXBW        m0, [dstq+strideq*0], %1 == 4
    PMOVZXBW        m1, [dstq+strideq*1], %1 == 4
    PMOVZXBW        m2, [dstq+strideq*2], %1 == 4
    PMOVZXBW        m3, [dstq+stride3q ], %1 == 4
 %if %2 == 8
    PMOVZXBW        m4, [dst4q+strideq*0], %1 == 4
    PMOVZXBW        m5, [dst4q+strideq*1], %1 == 4
    PMOVZXBW        m6, [dst4q+strideq*2], %1 == 4
    PMOVZXBW        m7, [dst4q+stride3q ], %1 == 4
 %endif
    mova     [px+0*%3], m0
    mova     [px+1*%3], m1
    mova     [px+2*%3], m2
    mova     [px+3*%3], m3
 %if %2 == 8
    mova     [px+4*%3], m4
    mova     [px+5*%3], m5
    mova     [px+6*%3], m6
    mova     [px+7*%3], m7
    mov dword [px+4*%3+%1*2], 0x7FFF7FFF
    mov dword [px+5*%3+%1*2], 0x7FFF7FFF
    mov dword [px+6*%3+%1*2], 0x7FFF7FFF
    mov dword [px+7*%3+%1*2], 0x7FFF7FFF
 %endif
    mov dword [px+0*%3+%1*2], 0x7FFF7FFF
    mov dword [px+1*%3+%1*2], 0x7FFF7FFF
    mov dword [px+2*%3+%1*2], 0x7FFF7FFF
    mov dword [px+3*%3+%1*2], 0x7FFF7FFF
.body_done:

    ; top
 %if ARCH_X86_64
    DEFINE_ARGS dst, stride, left, top2, pri, sec, stride3, top1, edge
 %else
    DEFINE_ARGS dst, stride, left, top2, stride3, top1, edge
 %endif
    LOAD_ARG32     top
    test         edged, 4                    ; have_top
    jz .no_top
    mov          top1q, [top2q+0*gprsize]
    mov          top2q, [top2q+1*gprsize]
    test         edged, 1                    ; have_left
    jz .top_no_left
    test         edged, 2                    ; have_right
    jz .top_no_right
 %if %1 == 4
    PMOVZXBW        m0, [top1q-2]
    PMOVZXBW        m1, [top2q-2]
 %else
    movu            m0, [top1q-4]
    movu            m1, [top2q-4]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    movu  [px-2*%3+8], m2
    movu  [px-1*%3+8], m3
 %endif
    movu  [px-2*%3-%1], m0
    movu  [px-1*%3-%1], m1
    jmp .top_done
.top_no_right:
 %if %1 == 4
    PMOVZXBW        m0, [top1q-%1]
    PMOVZXBW        m1, [top2q-%1]
    movu [px-2*%3-4*2], m0
    movu [px-1*%3-4*2], m1
 %else
    movu            m0, [top1q-%1]
    movu            m1, [top2q-%2]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    mova [px-2*%3-8*2], m0
    mova [px-2*%3-0*2], m2
    mova [px-1*%3-8*2], m1
    mova [px-1*%3-0*2], m3
 %endif
    mov dword [px-2*%3+%1*2], 0x7FFF7FFF
    mov dword [px-1*%3+%1*2], 0x7FFF7FFF
    jmp .top_done
.top_no_left:
    test         edged, 2                   ; have_right
    jz .top_no_left_right
 %if %1 == 4
    PMOVZXBW        m0, [top1q]
    PMOVZXBW        m1, [top2q]
 %else
    movu            m0, [top1q]
    movu            m1, [top2q]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    movd [px-2*%3+8*2], m2
    movd [px-1*%3+8*2], m3
 %endif
    mova     [px-2*%3], m0
    mova     [px-1*%3], m1
    mov dword [px-2*%3-4], 0x7FFF7FFF
    mov dword [px-1*%3-4], 0x7FFF7FFF
    jmp .top_done
.top_no_left_right:
    PMOVZXBW        m0, [top1q], %1 == 4
    PMOVZXBW        m1, [top2q], %1 == 4
    mova     [px-2*%3], m0
    mova     [px-1*%3], m1
    mov dword [px-2*%3+%1*2], 0x7FFF7FFF
    mov dword [px-1*%3+%1*2], 0x7FFF7FFF
    mov dword [px-2*%3-4], 0X7FFF7FFF
    mov dword [px-1*%3-4], 0X7FFF7FFF
    jmp .top_done
.no_top:
 %if ARCH_X86_64
    SWAP            m0, m14
 %else
    mova            m0, [PIC_sym(pw_0x7FFF)]
 %endif
    movu   [px-2*%3-4], m0
    movu   [px-1*%3-4], m0
 %if %1 == 8
    movq   [px-2*%3+12], m0
    movq   [px-1*%3+12], m0
 %endif
 %if ARCH_X86_64
    SWAP            m0, m14
 %endif
.top_done:

    ; left
    test         edged, 1                   ; have_left
    jz .no_left
    SAVE_PIC_REG     0
    LOAD_ARG32    left
 %if %2 == 4
    movq            m0, [leftq]
 %else
    movu            m0, [leftq]
 %endif
    LOAD_PIC_REG     0
 %if %2 == 4
    punpcklbw       m0, m15
 %else
    punpckhbw       m1, m0, m15
    punpcklbw       m0, m15
    movhlps         m3, m1
    movd   [px+4*%3-4], m1
    movd   [px+6*%3-4], m3
    psrlq           m1, 32
    psrlq           m3, 32
    movd   [px+5*%3-4], m1
    movd   [px+7*%3-4], m3
 %endif
    movhlps         m2, m0
    movd   [px+0*%3-4], m0
    movd   [px+2*%3-4], m2
    psrlq           m0, 32
    psrlq           m2, 32
    movd   [px+1*%3-4], m0
    movd   [px+3*%3-4], m2
    jmp .left_done
.no_left:
    mov dword [px+0*%3-4], 0x7FFF7FFF
    mov dword [px+1*%3-4], 0x7FFF7FFF
    mov dword [px+2*%3-4], 0x7FFF7FFF
    mov dword [px+3*%3-4], 0x7FFF7FFF
 %if %2 == 8
    mov dword [px+4*%3-4], 0x7FFF7FFF
    mov dword [px+5*%3-4], 0x7FFF7FFF
    mov dword [px+6*%3-4], 0x7FFF7FFF
    mov dword [px+7*%3-4], 0x7FFF7FFF
 %endif
.left_done:

    ; bottom
 %if ARCH_X86_64
    DEFINE_ARGS dst, stride, dummy1, dst8, pri, sec, stride3, dummy2, edge
 %else
    DEFINE_ARGS dst, stride, dummy1, dst8, stride3, dummy2, edge
 %endif
    test         edged, 8                   ; have_bottom
    jz .no_bottom
    lea          dst8q, [dstq+%2*strideq]
    test         edged, 1                   ; have_left
    jz .bottom_no_left
    test         edged, 2                   ; have_right
    jz .bottom_no_right
 %if %1 == 4
    PMOVZXBW        m0, [dst8q-(%1/2)]
    PMOVZXBW        m1, [dst8q+strideq-(%1/2)]
 %else
    movu            m0, [dst8q-4]
    movu            m1, [dst8q+strideq-4]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    movu [px+(%2+0)*%3+8], m2
    movu [px+(%2+1)*%3+8], m3
 %endif
    movu [px+(%2+0)*%3-%1], m0
    movu [px+(%2+1)*%3-%1], m1
    jmp .bottom_done
.bottom_no_right:
 %if %1 == 4
    PMOVZXBW        m0, [dst8q-4]
    PMOVZXBW        m1, [dst8q+strideq-4]
    movu [px+(%2+0)*%3-4*2], m0
    movu [px+(%2+1)*%3-4*2], m1
 %else
    movu            m0, [dst8q-8]
    movu            m1, [dst8q+strideq-8]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    mova [px+(%2+0)*%3-8*2], m0
    mova [px+(%2+0)*%3-0*2], m2
    mova [px+(%2+1)*%3-8*2], m1
    mova [px+(%2+1)*%3-0*2], m3
    mov dword [px+(%2-1)*%3+8*2], 0x7FFF7FFF    ; overwritten by first mova
 %endif
    mov dword [px+(%2+0)*%3+%1*2], 0x7FFF7FFF
    mov dword [px+(%2+1)*%3+%1*2], 0x7FFF7FFF
    jmp .bottom_done
.bottom_no_left:
    test          edged, 2                  ; have_right
    jz .bottom_no_left_right
 %if %1 == 4
    PMOVZXBW        m0, [dst8q]
    PMOVZXBW        m1, [dst8q+strideq]
 %else
    movu            m0, [dst8q]
    movu            m1, [dst8q+strideq]
    punpckhbw       m2, m0, m15
    punpcklbw       m0, m15
    punpckhbw       m3, m1, m15
    punpcklbw       m1, m15
    mova [px+(%2+0)*%3+8*2], m2
    mova [px+(%2+1)*%3+8*2], m3
 %endif
    mova [px+(%2+0)*%3], m0
    mova [px+(%2+1)*%3], m1
    mov dword [px+(%2+0)*%3-4], 0x7FFF7FFF
    mov dword [px+(%2+1)*%3-4], 0x7FFF7FFF
    jmp .bottom_done
.bottom_no_left_right:
    PMOVZXBW        m0, [dst8q+strideq*0], %1 == 4
    PMOVZXBW        m1, [dst8q+strideq*1], %1 == 4
    mova [px+(%2+0)*%3], m0
    mova [px+(%2+1)*%3], m1
    mov dword [px+(%2+0)*%3+%1*2], 0x7FFF7FFF
    mov dword [px+(%2+1)*%3+%1*2], 0x7FFF7FFF
    mov dword [px+(%2+0)*%3-4], 0x7FFF7FFF
    mov dword [px+(%2+1)*%3-4], 0x7FFF7FFF
    jmp .bottom_done
.no_bottom:
 %if ARCH_X86_64
    SWAP            m0, m14
 %else
    mova            m0, [PIC_sym(pw_0x7FFF)]
 %endif
    movu [px+(%2+0)*%3-4], m0
    movu [px+(%2+1)*%3-4], m0
 %if %1 == 8
    movq [px+(%2+0)*%3+12], m0
    movq [px+(%2+1)*%3+12], m0
 %endif
 %if ARCH_X86_64
    SWAP            m0, m14
 %endif
.bottom_done:

    ; actual filter
    DEFINE_ARGS dst, stride, pridmp, damping, pri, sec, secdmp
 %if ARCH_X86_64
    movifnidn     prid, prim
    movifnidn     secd, secm
    mov       dampingd, r7m
 %else
    LOAD_ARG       pri
    LOAD_ARG       sec
    LOAD_ARG   damping, 1
 %endif

    SAVE_PIC_REG     8
    mov        pridmpd, prid
    mov        secdmpd, secd
    or         pridmpd, 1
    or         secdmpd, 1
    bsr        pridmpd, pridmpd
    bsr        secdmpd, secdmpd
    sub        pridmpd, dampingd
    sub        secdmpd, dampingd
    xor       dampingd, dampingd
    neg        pridmpd
    cmovl      pridmpd, dampingd
    neg        secdmpd
    cmovl      secdmpd, dampingd
    mov       [rsp+ 0], pridmpq                 ; pri_shift
    mov       [rsp+16], secdmpq                 ; sec_shift
 %if ARCH_X86_32
    mov dword [esp+ 4], 0                       ; zero upper 32 bits of psraw
    mov dword [esp+20], 0                       ; source operand in ACCUMULATE_TAP
  %define PIC_reg r6
    LOAD_PIC_REG     8
 %endif

    ; pri/sec_taps[k] [4 total]
    DEFINE_ARGS dst, stride, tap, dummy, pri, sec
 %if ARCH_X86_64
    mova           m14, [pw_256]
 %else
  %define m14   [PIC_sym(pw_256)]
 %endif
    movd            m0, prid
    movd            m1, secd
    pshufb          m0, m14
    pshufb          m1, m14
 %if ARCH_X86_32
    mova            m2, [PIC_sym(pw_0x7FFF)]
    pandn           m0, m2
    pandn           m1, m2
    mova    [esp+0x20], m0
    mova    [esp+0x30], m1
 %endif
    and           prid, 1
    lea           tapq, [PIC_sym(tap_table)]
    lea           priq, [tapq+priq*4]           ; pri_taps
    lea           secq, [tapq+8]                ; sec_taps

    ; off1/2/3[k] [6 total] from [tapq+12+(dir+0/2/6)*2+k]
    DEFINE_ARGS dst, stride, tap, dir, pri, sec
 %if ARCH_X86_64
    mov           dird, r6m
    lea           tapq, [tapq+dirq*2+12]
    DEFINE_ARGS dst, stride, dir, stk, pri, sec, h, off, k
 %else
    LOAD_ARG       dir, 1
    lea           tapd, [tapd+dird*2+12]
    DEFINE_ARGS dst, stride, dir, stk, pri, sec
  %define hd    dword [esp+8]
  %define offq  dstq
  %define kq    strideq
 %endif
    mov             hd, %1*%2*2/mmsize
    lea           stkq, [px]
    movif32 [esp+0x1C], strided
.v_loop:
    movif32 [esp+0x18], dstd
    mov             kq, 1
 %if %1 == 4
    movq            m4, [stkq+%3*0]
    movhps          m4, [stkq+%3*1]
 %else
    mova            m4, [stkq+%3*0]             ; px
 %endif

 %if ARCH_X86_32
  %xdefine m11  m6
  %xdefine m13  m7
  %xdefine  m7  m0
  %xdefine  m8  m1
 %endif

    pxor           m13, m13                     ; sum
    mova            m7, m4                      ; max
    mova            m8, m4                      ; min
.k_loop:
 %if ARCH_X86_64
    movd            m2, [priq+kq*2]             ; pri_taps
    movd            m3, [secq+kq*2]             ; sec_taps
    pshufb          m2, m14
    pshufb          m3, m14
    ACCUMULATE_TAP 0*2, [rsp+ 0], m0, m2, %1, %3
    ACCUMULATE_TAP 2*2, [rsp+16], m1, m3, %1, %3
    ACCUMULATE_TAP 6*2, [rsp+16], m1, m3, %1, %3
 %else
    movd            m2, [priq+kq*2]             ; pri_taps
    pshufb          m2, m14
    ACCUMULATE_TAP 0*2, [esp+0x00], [esp+0x20], m2, %1, %3

    movd            m2, [secq+kq*2]             ; sec_taps
    pshufb          m2, m14
    ACCUMULATE_TAP 2*2, [esp+0x10], [esp+0x30], m2, %1, %3
    ACCUMULATE_TAP 6*2, [esp+0x10], [esp+0x30], m2, %1, %3
 %endif

    dec             kq
    jge .k_loop

    pcmpgtw        m11, m15, m13
    paddw          m13, m11
    pmulhrsw       m13, [PIC_sym(pw_2048)]
    paddw           m4, m13
    pminsw          m4, m7
    pmaxsw          m4, m8
    packuswb        m4, m4
    movif32       dstd, [esp+0x18]
    movif32    strided, [esp+0x1C]
 %if %1 == 4
    movd [dstq+strideq*0], m4
    psrlq           m4, 32
    movd [dstq+strideq*1], m4
 %else
    movq [dstq], m4
 %endif

 %if %1 == 4
 %define vloop_lines (mmsize/(%1*2))
    lea           dstq, [dstq+strideq*vloop_lines]
    add           stkq, %3*vloop_lines
 %else
    lea           dstq, [dstq+strideq]
    add           stkq, %3
 %endif
    dec             hd
    jg .v_loop

    RET
%endmacro

cdef_filter_fn 8, 8, 32
cdef_filter_fn 4, 8, 32
cdef_filter_fn 4, 4, 32
