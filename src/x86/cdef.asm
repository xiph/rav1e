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

%if ARCH_X86_64

%macro DUP4 1-*
    %rep %0
        times 4 db %1
        %rotate 1
    %endrep
%endmacro

%macro DIRS 16 ; cdef_directions[]
    %rep 4 + 16 + 4 ; 6 7   0 1 2 3 4 5 6 7   0 1
        ; masking away unused bits allows us to use a single vpaddd {1to16}
        ; instruction instead of having to do vpbroadcastd + paddb
        db %13 & 0x3f, -%13 & 0x3f
        %rotate 1
    %endrep
%endmacro

%macro JMP_TABLE 2-*
 %xdefine %1_jmptable %%table
 %xdefine %%base mangle(private_prefix %+ _%1_avx2)
 %%table:
 %rep %0 - 1
    dd %%base %+ .%2 - %%table
  %rotate 1
 %endrep
%endmacro

%macro CDEF_FILTER_JMP_TABLE 1
JMP_TABLE cdef_filter_%1, \
    d6k0, d6k1, d7k0, d7k1, \
    d0k0, d0k1, d1k0, d1k1, d2k0, d2k1, d3k0, d3k1, \
    d4k0, d4k1, d5k0, d5k1, d6k0, d6k1, d7k0, d7k1, \
    d0k0, d0k1, d1k0, d1k1
%endmacro

SECTION_RODATA 64

lut_perm_4x4:  db 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
               db 16, 17,  0,  1,  2,  3,  4,  5, 18, 19,  8,  9, 10, 11, 12, 13
               db 20, 21, 80, 81, 82, 83, 84, 85, 22, 23, 32, 33, 34, 35, 36, 37
               db 98, 99,100,101,102,103,104,105, 50, 51, 52, 53, 54, 55, 56, 57
lut_perm_4x8a: db 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
              db  96, 97,  0,  1,  2,  3,  4,  5, 98, 99,  8,  9, 10, 11, 12, 13
lut_perm_4x8b:db 100,101, 16, 17, 18, 19, 20, 21,102,103, 24, 25, 26, 27, 28, 29
              db 104,105, 32, 33, 34, 35, 36, 37,106,107, 40, 41, 42, 43, 44, 45
              db 108,109, 48, 49, 50, 51, 52, 53,110,111, 56, 57, 58, 59, 60, 61
               db 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
pd_01234567:   dd  0,  1,  2,  3,  4,  5,  6,  7
lut_perm_8x8a: db  0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23
               db -1, -1, 34, 35, 36, 37, 38, 39, -1, -1, 50, 51, 52, 53, 54, 55
               db -1, -1, 66, 67, 68, 69, 70, 71, -1, -1, 82, 83, 84, 85, 86, 87
               db 96, 97, 98, 99,100,101,102,103,112,113,114,115,116,117,118,119
lut_perm_8x8b: db  4,  5,  6,  7,  8,  9, 10, 11, 20, 21, 22, 23, 24, 25, 26, 27
               db 36, 37, 38, 39, 40, 41, 42, 43, 52, 53, 54, 55, 56, 57, 58, 59
               db 68, 69, 70, 71, 72, 73, 74, 75, 84, 85, 86, 87, 88, 89, 90, 91
              db 100,101,102,103,104,105,106,107,116,117,118,119,120,121,122,123
edge_mask:     dq 0x00003c3c3c3c0000, 0x00003f3f3f3f0000 ; 0000, 0001
               dq 0x0000fcfcfcfc0000, 0x0000ffffffff0000 ; 0010, 0011
               dq 0x00003c3c3c3c3c3c, 0x00003f3f3f3f3f3f ; 0100, 0101
               dq 0x0000fcfcfcfcfcfc, 0x0000ffffffffffff ; 0110, 0111
               dq 0x3c3c3c3c3c3c0000, 0x3f3f3f3f3f3f0000 ; 1000, 1001
               dq 0xfcfcfcfcfcfc0000, 0xffffffffffff0000 ; 1010, 1011
               dq 0x3c3c3c3c3c3c3c3c, 0x3f3f3f3f3f3f3f3f ; 1100, 1101
               dq 0xfcfcfcfcfcfcfcfc, 0xffffffffffffffff ; 1110, 1111
px_idx:      DUP4 18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45
cdef_dirs:   DIRS -7,-14,  1, -6,  1,  2,  1, 10,  9, 18,  8, 17,  8, 16,  8, 15
gf_shr:        dq 0x0102040810204080, 0x0102040810204080 ; >> 0, >> 0
               dq 0x0204081020408000, 0x0408102040800000 ; >> 1, >> 2
               dq 0x0810204080000000, 0x1020408000000000 ; >> 3, >> 4
               dq 0x2040800000000000, 0x4080000000000000 ; >> 5, >> 6
      times 16 db  0 ; realign (introduced by cdef_dirs)
end_perm_w8clip:db 0, 4,  8, 12,  2,  6, 10, 14, 16, 20, 24, 28, 18, 22, 26, 30
               db 32, 36, 40, 44, 34, 38, 42, 46, 48, 52, 56, 60, 50, 54, 58, 62
               db  1,  5,  9, 13,  3,  7, 11, 15, 17, 21, 25, 29, 19, 23, 27, 31
               db 33, 37, 41, 45, 35, 39, 43, 47, 49, 53, 57, 61, 51, 55, 59, 63
end_perm:      db  1,  5,  9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61
               db  3,  7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
pri_tap:       db 64, 64, 32, 32, 48, 48, 48, 48         ; left-shifted by 4
sec_tap:       db 32, 32, 16, 16
pd_268435568:  dd 268435568
blend_4x4:     dd 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00
               dd 0x80, 0x00, 0x00
blend_4x8_0:   dd 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
blend_4x8_1:   dd 0x00, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
               dd 0x00, 0x00
blend_4x8_2:   dd 0x0000, 0x8080, 0x8080, 0x8080, 0x8080, 0x8080, 0x8080, 0x8080
               dd 0x0000
blend_4x8_3:   dd 0x0000, 0x0000, 0x8080, 0x8080, 0x8080, 0x8080, 0x8080, 0x8080
               dd 0x0000, 0x0000
blend_8x8_0:   dq 0x00, 0x00, 0x80, 0x80, 0x80, 0x80
blend_8x8_1:   dq 0x0000, 0x0000, 0x8080, 0x8080, 0x8080, 0x8080, 0x0000, 0x0000
pd_47130256:   dd  4,  7,  1,  3,  0,  2,  5,  6
div_table:     dd 840, 420, 280, 210, 168, 140, 120, 105, 420, 210, 140, 105
shufw_6543210x:db 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1, 14, 15
shufb_lohi:    db  0,  8,  1,  9,  2, 10,  3, 11,  4, 12,  5, 13,  6, 14,  7, 15
pw_128:        times 2 dw 128
pw_2048:       times 2 dw 2048
tap_table:     ; masks for 8 bit shifts
               db 0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01
               ; weights
               db  4,  2,  3,  3,  2,  1
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

CDEF_FILTER_JMP_TABLE 4x4
CDEF_FILTER_JMP_TABLE 4x8
CDEF_FILTER_JMP_TABLE 8x8

SECTION .text

%macro PREP_REGS 2 ; w, h
    ; off1/2/3[k] [6 total] from [tapq+12+(dir+0/2/6)*2+k]
    mov           dird, r6m
    lea         tableq, [cdef_filter_%1x%2_jmptable]
    lea           dirq, [tableq+dirq*2*4]
%if %1 == 4
 %if %2 == 4
  DEFINE_ARGS dst, stride, left, top, pri, sec, \
              table, dir, dirjmp, dst4, stride3, k
 %else
  DEFINE_ARGS dst, stride, left, top, pri, sec, \
              table, dir, dirjmp, dst4, dst8, stride3, k
    lea          dst8q, [dstq+strideq*8]
 %endif
%else
  DEFINE_ARGS dst, stride, h, top1, pri, sec, \
              table, dir, dirjmp, top2, dst4, stride3, k
    mov             hq, -8
    lea          top1q, [top1q+strideq*0]
    lea          top2q, [top1q+strideq*1]
%endif
    lea          dst4q, [dstq+strideq*4]
%if %1 == 4
    lea       stride3q, [strideq*3]
%endif
%endmacro

%macro LOAD_BLOCK 2-3 0 ; w, h, init_min_max
    mov             kd, 1
    pxor           m15, m15                     ; sum
%if %2 == 8
    pxor           m12, m12
 %if %1 == 4
    movd           xm4, [dstq +strideq*0]
    movd           xm6, [dstq +strideq*1]
    movd           xm5, [dstq +strideq*2]
    movd           xm7, [dstq +stride3q ]
    vinserti128     m4, [dst4q+strideq*0], 1
    vinserti128     m6, [dst4q+strideq*1], 1
    vinserti128     m5, [dst4q+strideq*2], 1
    vinserti128     m7, [dst4q+stride3q ], 1
    punpckldq       m4, m6
    punpckldq       m5, m7
 %else
    movq           xm4, [dstq+strideq*0]
    movq           xm5, [dstq+strideq*1]
    vinserti128     m4, [dstq+strideq*2], 1
    vinserti128     m5, [dstq+stride3q ], 1
 %endif
    punpcklqdq      m4, m5
%else
    movd           xm4, [dstq+strideq*0]
    movd           xm5, [dstq+strideq*1]
    vinserti128     m4, [dstq+strideq*2], 1
    vinserti128     m5, [dstq+stride3q ], 1
    punpckldq       m4, m5
%endif
%if %3 == 1
    mova            m7, m4                      ; min
    mova            m8, m4                      ; max
%endif
%endmacro

%macro ACCUMULATE_TAP_BYTE 7-8 0 ; tap_offset, shift, mask, strength
                                 ; mul_tap, w, h, clip
    ; load p0/p1
    movsxd     dirjmpq, [dirq+kq*4+%1*2*4]
    add        dirjmpq, tableq
    call       dirjmpq

%if %8 == 1
    pmaxub          m7, m5
    pminub          m8, m5
    pmaxub          m7, m6
    pminub          m8, m6
%endif

    ; accumulate sum[m15] over p0/p1
%if %7 == 4
    punpcklbw       m5, m6
    punpcklbw       m6, m4, m4
    psubusb         m9, m5, m6
    psubusb         m5, m6, m5
    por             m9, m5     ; abs_diff_p01(p01 - px)
    pcmpeqb         m5, m9
    por             m5, %5
    psignb          m6, %5, m5
    psrlw           m5, m9, %2 ; emulate 8-bit shift
    pand            m5, %3
    psubusb         m5, %4, m5
    pminub          m5, m9
    pmaddubsw       m5, m6
    paddw          m15, m5
%else
    psubusb         m9, m5, m4
    psubusb         m5, m4, m5
    psubusb        m11, m6, m4
    psubusb         m6, m4, m6
    por             m9, m5      ; abs_diff_p0(p0 - px)
    por            m11, m6      ; abs_diff_p1(p1 - px)
    pcmpeqb         m5, m9
    pcmpeqb         m6, m11
    punpckhbw      m10, m9, m11
    punpcklbw       m9, m11
    por             m5, %5
    por            m11, m6, %5
    punpckhbw       m6, m5, m11
    punpcklbw       m5, m11
    psignb         m11, %5, m6
    psrlw           m6, m10, %2 ; emulate 8-bit shift
    pand            m6, %3
    psubusb         m6, %4, m6
    pminub          m6, m10
    pmaddubsw       m6, m11
    paddw          m12, m6
    psignb         m11, %5, m5
    psrlw           m5, m9, %2  ; emulate 8-bit shift
    pand            m5, %3
    psubusb         m5, %4, m5
    pminub          m5, m9
    pmaddubsw       m5, m11
    paddw          m15, m5
%endif
%endmacro

%macro ADJUST_PIXEL 4-5 0 ; w, h, zero, pw_2048, clip
%if %2 == 4
 %if %5 == 1
    punpcklbw       m4, %3
 %endif
    pcmpgtw         %3, m15
    paddw          m15, %3
    pmulhrsw       m15, %4
 %if %5 == 0
    packsswb       m15, m15
    paddb           m4, m15
 %else
    paddw           m4, m15
    packuswb        m4, m4 ; clip px in [0x0,0xff]
    pminub          m4, m7
    pmaxub          m4, m8
 %endif
    vextracti128   xm5, m4, 1
    movd   [dstq+strideq*0], xm4
    movd   [dstq+strideq*2], xm5
    pextrd [dstq+strideq*1], xm4, 1
    pextrd [dstq+stride3q ], xm5, 1
%else
    pcmpgtw         m6, %3, m12
    pcmpgtw         m5, %3, m15
    paddw          m12, m6
    paddw          m15, m5
 %if %5 == 1
    punpckhbw       m5, m4, %3
    punpcklbw       m4, %3
 %endif
    pmulhrsw       m12, %4
    pmulhrsw       m15, %4
 %if %5 == 0
    packsswb       m15, m12
    paddb           m4, m15
 %else
    paddw           m5, m12
    paddw           m4, m15
    packuswb        m4, m5 ; clip px in [0x0,0xff]
    pminub          m4, m7
    pmaxub          m4, m8
 %endif
    vextracti128   xm5, m4, 1
 %if %1 == 4
    movd   [dstq +strideq*0], xm4
    movd   [dst4q+strideq*0], xm5
    pextrd [dstq +strideq*1], xm4, 1
    pextrd [dst4q+strideq*1], xm5, 1
    pextrd [dstq +strideq*2], xm4, 2
    pextrd [dst4q+strideq*2], xm5, 2
    pextrd [dstq +stride3q ], xm4, 3
    pextrd [dst4q+stride3q ], xm5, 3
 %else
    movq   [dstq+strideq*0], xm4
    movq   [dstq+strideq*2], xm5
    movhps [dstq+strideq*1], xm4
    movhps [dstq+stride3q ], xm5
 %endif
%endif
%endmacro

%macro BORDER_PREP_REGS 2 ; w, h
    ; off1/2/3[k] [6 total] from [tapq+12+(dir+0/2/6)*2+k]
    mov           dird, r6m
    lea           dirq, [tableq+dirq*2+14]
%if %1*%2*2/mmsize > 1
 %if %1 == 4
    DEFINE_ARGS dst, stride, dir, stk, pri, sec, stride3, h, off, k
 %else
    DEFINE_ARGS dst, stride, dir, stk, pri, sec, h, off, k
 %endif
    mov             hd, %1*%2*2/mmsize
%else
    DEFINE_ARGS dst, stride, dir, stk, pri, sec, stride3, off, k
%endif
    lea           stkq, [px]
    pxor           m11, m11
%endmacro

%macro BORDER_LOAD_BLOCK 2-3 0 ; w, h, init_min_max
    mov             kd, 1
%if %1 == 4
    movq           xm4, [stkq+32*0]
    movhps         xm4, [stkq+32*1]
    movq           xm5, [stkq+32*2]
    movhps         xm5, [stkq+32*3]
    vinserti128     m4, xm5, 1
%else
    mova           xm4, [stkq+32*0]             ; px
    vinserti128     m4, [stkq+32*1], 1
%endif
    pxor           m15, m15                     ; sum
%if %3 == 1
    mova            m7, m4                      ; max
    mova            m8, m4                      ; min
%endif
%endmacro

%macro ACCUMULATE_TAP_WORD 6-7 0 ; tap_offset, shift, mask, strength
                                 ; mul_tap, w, clip
    ; load p0/p1
    movsx         offq, byte [dirq+kq+%1]       ; off1
%if %6 == 4
    movq           xm5, [stkq+offq*2+32*0]      ; p0
    movq           xm6, [stkq+offq*2+32*2]
    movhps         xm5, [stkq+offq*2+32*1]
    movhps         xm6, [stkq+offq*2+32*3]
    vinserti128     m5, xm6, 1
%else
    movu           xm5, [stkq+offq*2+32*0]      ; p0
    vinserti128     m5, [stkq+offq*2+32*1], 1
%endif
    neg           offq                          ; -off1
%if %6 == 4
    movq           xm6, [stkq+offq*2+32*0]      ; p1
    movq           xm9, [stkq+offq*2+32*2]
    movhps         xm6, [stkq+offq*2+32*1]
    movhps         xm9, [stkq+offq*2+32*3]
    vinserti128     m6, xm9, 1
%else
    movu           xm6, [stkq+offq*2+32*0]      ; p1
    vinserti128     m6, [stkq+offq*2+32*1], 1
%endif
%if %7 == 1
    ; out of bounds values are set to a value that is a both a large unsigned
    ; value and a negative signed value.
    ; use signed max and unsigned min to remove them
    pmaxsw          m7, m5                      ; max after p0
    pminuw          m8, m5                      ; min after p0
    pmaxsw          m7, m6                      ; max after p1
    pminuw          m8, m6                      ; min after p1
%endif

    ; accumulate sum[m15] over p0/p1
    ; calculate difference before converting
    psubw           m5, m4                      ; diff_p0(p0 - px)
    psubw           m6, m4                      ; diff_p1(p1 - px)

    ; convert to 8-bits with signed saturation
    ; saturating to large diffs has no impact on the results
    packsswb        m5, m6

    ; group into pairs so we can accumulate using maddubsw
    pshufb          m5, m12
    pabsb           m9, m5
    psignb         m10, %5, m5
    psrlw           m5, m9, %2                  ; emulate 8-bit shift
    pand            m5, %3
    psubusb         m5, %4, m5

    ; use unsigned min since abs diff can equal 0x80
    pminub          m5, m9
    pmaddubsw       m5, m10
    paddw          m15, m5
%endmacro

%macro BORDER_ADJUST_PIXEL 2-3 0 ; w, pw_2048, clip
    pcmpgtw         m9, m11, m15
    paddw          m15, m9
    pmulhrsw       m15, %2
    paddw           m4, m15
%if %3 == 1
    pminsw          m4, m7
    pmaxsw          m4, m8
%endif
    packuswb        m4, m4
    vextracti128   xm5, m4, 1
%if %1 == 4
    movd [dstq+strideq*0], xm4
    pextrd [dstq+strideq*1], xm4, 1
    movd [dstq+strideq*2], xm5
    pextrd [dstq+stride3q], xm5, 1
%else
    movq [dstq+strideq*0], xm4
    movq [dstq+strideq*1], xm5
%endif
%endmacro

%macro CDEF_FILTER 2 ; w, h
INIT_YMM avx2
cglobal cdef_filter_%1x%2, 4, 9, 0, dst, stride, left, top, \
                                    pri, sec, dir, damping, edge
%assign stack_offset_entry stack_offset
    mov          edged, edgem
    cmp          edged, 0xf
    jne .border_block

    PUSH            r9
    PUSH           r10
    PUSH           r11
%if %2 == 4
 %assign regs_used 12
 %if STACK_ALIGNMENT < 32
    PUSH  r%+regs_used
  %assign regs_used regs_used+1
 %endif
    ALLOC_STACK 0x60, 16
    pmovzxbw       xm0, [leftq+1]
    vpermq          m0, m0, q0110
    psrldq          m1, m0, 4
    vpalignr        m2, m0, m0, 12
    movu    [rsp+0x10], m0
    movu    [rsp+0x28], m1
    movu    [rsp+0x40], m2
%elif %1 == 4
    PUSH           r12
 %assign regs_used 13
 %if STACK_ALIGNMENT < 32
    PUSH  r%+regs_used
   %assign regs_used regs_used+1
 %endif
    ALLOC_STACK 8*2+%1*%2*1, 16
    pmovzxwd        m0, [leftq]
    mova    [rsp+0x10], m0
%else
    PUSH           r12
    PUSH           r13
 %assign regs_used 14
 %if STACK_ALIGNMENT < 32
    PUSH  r%+regs_used
  %assign regs_used regs_used+1
 %endif
    ALLOC_STACK 8*2+%1*%2*2+32, 16
    lea            r11, [strideq*3]
    movu           xm4, [dstq+strideq*2]
    pmovzxwq        m0, [leftq+0]
    pmovzxwq        m1, [leftq+8]
    vinserti128     m4, [dstq+r11], 1
    pmovzxbd        m2, [leftq+1]
    pmovzxbd        m3, [leftq+9]
    mova    [rsp+0x10], m0
    mova    [rsp+0x30], m1
    mova    [rsp+0x50], m2
    mova    [rsp+0x70], m3
    mova    [rsp+0x90], m4
%endif

 DEFINE_ARGS dst, stride, left, top, pri, secdmp, zero, pridmp, damping
    mov       dampingd, r7m
    xor          zerod, zerod
    movifnidn     prid, prim
    sub       dampingd, 31
    movifnidn  secdmpd, secdmpm
    or            prid, 0
    jz .sec_only
    movd           xm0, prid
    lzcnt      pridmpd, prid
    add        pridmpd, dampingd
    cmovs      pridmpd, zerod
    mov        [rsp+0], pridmpq                 ; pri_shift
    or         secdmpd, 0
    jz .pri_only
    movd           xm1, secdmpd
    lzcnt      secdmpd, secdmpd
    add        secdmpd, dampingd
    cmovs      secdmpd, zerod
    mov        [rsp+8], secdmpq                 ; sec_shift

 DEFINE_ARGS dst, stride, left, top, pri, secdmp, table, pridmp
    lea         tableq, [tap_table]
    vpbroadcastb   m13, [tableq+pridmpq]        ; pri_shift_mask
    vpbroadcastb   m14, [tableq+secdmpq]        ; sec_shift_mask

    ; pri/sec_taps[k] [4 total]
 DEFINE_ARGS dst, stride, left, top, pri, sec, table, dir
    vpbroadcastb    m0, xm0                     ; pri_strength
    vpbroadcastb    m1, xm1                     ; sec_strength
    and           prid, 1
    lea           priq, [tableq+priq*2+8]       ; pri_taps
    lea           secq, [tableq+12]             ; sec_taps

    PREP_REGS       %1, %2
%if %1*%2 > mmsize
.v_loop:
%endif
    LOAD_BLOCK      %1, %2, 1
.k_loop:
    vpbroadcastb    m2, [priq+kq]                          ; pri_taps
    vpbroadcastb    m3, [secq+kq]                          ; sec_taps
    ACCUMULATE_TAP_BYTE 2, [rsp+0], m13, m0, m2, %1, %2, 1 ; dir + 0
    ACCUMULATE_TAP_BYTE 4, [rsp+8], m14, m1, m3, %1, %2, 1 ; dir + 2
    ACCUMULATE_TAP_BYTE 0, [rsp+8], m14, m1, m3, %1, %2, 1 ; dir - 2
    dec             kq
    jge .k_loop

    vpbroadcastd   m10, [pw_2048]
    pxor            m9, m9
    ADJUST_PIXEL    %1, %2, m9, m10, 1
%if %1*%2 > mmsize
    mov           dstq, dst4q
    lea          top1q, [rsp+0x90]
    lea          top2q, [rsp+0xA0]
    lea          dst4q, [dst4q+strideq*4]
    add             hq, 4
    jl .v_loop
%endif
    RET

.pri_only:
 DEFINE_ARGS dst, stride, left, top, pri, _, table, pridmp
    lea         tableq, [tap_table]
    vpbroadcastb   m13, [tableq+pridmpq]        ; pri_shift_mask
    ; pri/sec_taps[k] [4 total]
 DEFINE_ARGS dst, stride, left, top, pri, _, table, dir
    vpbroadcastb    m0, xm0                     ; pri_strength
    and           prid, 1
    lea           priq, [tableq+priq*2+8]       ; pri_taps
    PREP_REGS       %1, %2
    vpbroadcastd    m3, [pw_2048]
    pxor            m1, m1
%if %1*%2 > mmsize
.pri_v_loop:
%endif
    LOAD_BLOCK      %1, %2
.pri_k_loop:
    vpbroadcastb    m2, [priq+kq]                       ; pri_taps
    ACCUMULATE_TAP_BYTE 2, [rsp+0], m13, m0, m2, %1, %2 ; dir + 0
    dec             kq
    jge .pri_k_loop
    ADJUST_PIXEL    %1, %2, m1, m3
%if %1*%2 > mmsize
    mov           dstq, dst4q
    lea          top1q, [rsp+0x90]
    lea          top2q, [rsp+0xA0]
    lea          dst4q, [dst4q+strideq*4]
    add             hq, 4
    jl .pri_v_loop
%endif
    RET

.sec_only:
 DEFINE_ARGS dst, stride, left, top, _, secdmp, zero, _, damping
    movd           xm1, secdmpd
    lzcnt      secdmpd, secdmpd
    add        secdmpd, dampingd
    cmovs      secdmpd, zerod
    mov        [rsp+8], secdmpq                 ; sec_shift
 DEFINE_ARGS dst, stride, left, top, _, secdmp, table
    lea         tableq, [tap_table]
    vpbroadcastb   m14, [tableq+secdmpq]        ; sec_shift_mask
    ; pri/sec_taps[k] [4 total]
 DEFINE_ARGS dst, stride, left, top, _, sec, table, dir
    vpbroadcastb    m1, xm1                     ; sec_strength
    lea           secq, [tableq+12]             ; sec_taps
    PREP_REGS       %1, %2
    vpbroadcastd    m2, [pw_2048]
    pxor            m0, m0
%if %1*%2 > mmsize
.sec_v_loop:
%endif
    LOAD_BLOCK      %1, %2
.sec_k_loop:
    vpbroadcastb    m3, [secq+kq]                       ; sec_taps
    ACCUMULATE_TAP_BYTE 4, [rsp+8], m14, m1, m3, %1, %2 ; dir + 2
    ACCUMULATE_TAP_BYTE 0, [rsp+8], m14, m1, m3, %1, %2 ; dir - 2
    dec             kq
    jge .sec_k_loop
    ADJUST_PIXEL    %1, %2, m0, m2
%if %1*%2 > mmsize
    mov           dstq, dst4q
    lea          top1q, [rsp+0x90]
    lea          top2q, [rsp+0xA0]
    lea          dst4q, [dst4q+strideq*4]
    add             hq, 4
    jl .sec_v_loop
%endif
    RET

.d0k0:
%if %1 == 4
 %if %2 == 4
    vpbroadcastq    m6, [dstq+strideq*1-1]
    vpbroadcastq   m10, [dstq+strideq*2-1]
    movd           xm5, [topq+strideq*1+1]
    movd           xm9, [dstq+strideq*0+1]
    psrldq         m11, m6, 2
    psrldq         m12, m10, 2
    vinserti128     m6, [dstq+stride3q -1], 1
    vinserti128    m10, [dstq+strideq*4-1], 1
    vpblendd        m5, m11, 0x10
    vpblendd        m9, m12, 0x10
    movu           m11, [blend_4x4+16]
    punpckldq       m6, m10
    punpckldq       m5, m9
    vpblendvb       m6, [rsp+gprsize+0x28], m11
 %else
    movd           xm5, [topq +strideq*1+1]
    movq           xm6, [dstq +strideq*1-1]
    movq          xm10, [dstq +stride3q -1]
    movq          xm11, [dst4q+strideq*1-1]
    pinsrd         xm5, [dstq +strideq*0+1], 1
    movhps         xm6, [dstq +strideq*2-1]
    movhps        xm10, [dst4q+strideq*0-1]
    movhps        xm11, [dst4q+strideq*2-1]
    psrldq         xm9, xm6, 2
    shufps         xm5, xm9, q2010   ; -1 +0 +1 +2
    shufps         xm6, xm10, q2020  ; +1 +2 +3 +4
    psrldq         xm9, xm11, 2
    psrldq        xm10, 2
    shufps        xm10, xm9, q2020   ; +3 +4 +5 +6
    movd           xm9, [dst4q+stride3q -1]
    pinsrd         xm9, [dst4q+strideq*4-1], 1
    shufps        xm11, xm9, q1020   ; +5 +6 +7 +8
    pmovzxbw        m9, [leftq+3]
    vinserti128     m6, xm11, 1
    movu           m11, [blend_4x8_0+4]
    vinserti128     m5, xm10, 1
    vpblendvb       m6, m9, m11
 %endif
%else
    lea            r13, [blend_8x8_0+16]
    movq           xm5, [top2q         +1]
    vbroadcasti128 m10, [dstq+strideq*1-1]
    vbroadcasti128 m11, [dstq+strideq*2-1]
    movhps         xm5, [dstq+strideq*0+1]
    vinserti128     m6, m10, [dstq+stride3q -1], 1
    vinserti128     m9, m11, [dstq+strideq*4-1], 1
    psrldq         m10, 2
    psrldq         m11, 2
    punpcklqdq      m6, m9
    movu            m9, [r13+hq*2*1+16*1]
    punpcklqdq     m10, m11
    vpblendd        m5, m10, 0xF0
    vpblendvb       m6, [rsp+gprsize+80+hq*8+64+8*1], m9
%endif
    ret
.d1k0:
.d2k0:
.d3k0:
%if %1 == 4
 %if %2 == 4
    movq           xm6, [dstq+strideq*0-1]
    movq           xm9, [dstq+strideq*1-1]
    vinserti128     m6, [dstq+strideq*2-1], 1
    vinserti128     m9, [dstq+stride3q -1], 1
    movu           m11, [rsp+gprsize+0x10]
    pcmpeqd        m12, m12
    psrldq          m5, m6, 2
    psrldq         m10, m9, 2
    psrld          m12, 24
    punpckldq       m6, m9
    punpckldq       m5, m10
    vpblendvb       m6, m11, m12
 %else
    movq           xm6, [dstq +strideq*0-1]
    movq           xm9, [dstq +strideq*2-1]
    movhps         xm6, [dstq +strideq*1-1]
    movhps         xm9, [dstq +stride3q -1]
    movq          xm10, [dst4q+strideq*0-1]
    movhps        xm10, [dst4q+strideq*1-1]
    psrldq         xm5, xm6, 2
    psrldq        xm11, xm9, 2
    shufps         xm5, xm11, q2020
    movq          xm11, [dst4q+strideq*2-1]
    movhps        xm11, [dst4q+stride3q -1]
    shufps         xm6, xm9, q2020
    shufps         xm9, xm10, xm11, q2020
    vinserti128     m6, xm9, 1
    pmovzxbw        m9, [leftq+1]
    psrldq        xm10, 2
    psrldq        xm11, 2
    shufps        xm10, xm11, q2020
    vpbroadcastd   m11, [blend_4x8_0+4]
    vinserti128     m5, xm10, 1
    vpblendvb       m6, m9, m11
 %endif
%else
    movu           xm5, [dstq+strideq*0-1]
    movu           xm9, [dstq+strideq*1-1]
    vinserti128     m5, [dstq+strideq*2-1], 1
    vinserti128     m9, [dstq+stride3q -1], 1
    mova           m10, [blend_8x8_0+16]
    punpcklqdq      m6, m5, m9
    vpblendvb       m6, [rsp+gprsize+80+hq*8+64], m10
    psrldq          m5, 2
    psrldq          m9, 2
    punpcklqdq      m5, m9
%endif
    ret
.d4k0:
%if %1 == 4
 %if %2 == 4
    vpbroadcastq   m10, [dstq+strideq*1-1]
    vpbroadcastq   m11, [dstq+strideq*2-1]
    movd           xm6, [topq+strideq*1-1]
    movd           xm9, [dstq+strideq*0-1]
    psrldq          m5, m10, 2
    psrldq         m12, m11, 2
    vpblendd        m6, m10, 0x10
    vpblendd        m9, m11, 0x10
    movu           m10, [blend_4x4]
    vinserti128     m5, [dstq+stride3q +1], 1
    vinserti128    m12, [dstq+strideq*4+1], 1
    punpckldq       m6, m9
    punpckldq       m5, m12
    vpblendvb       m6, [rsp+gprsize+0x40], m10
 %else
    movd           xm6, [topq +strideq*1-1]
    movq           xm9, [dstq +strideq*1-1]
    movq          xm10, [dstq +stride3q -1]
    movq          xm11, [dst4q+strideq*1-1]
    pinsrd         xm6, [dstq +strideq*0-1], 1
    movhps         xm9, [dstq +strideq*2-1]
    movhps        xm10, [dst4q+strideq*0-1]
    movhps        xm11, [dst4q+strideq*2-1]
    psrldq         xm5, xm9, 2
    shufps         xm6, xm9, q2010
    psrldq         xm9, xm10, 2
    shufps         xm5, xm9, q2020
    shufps        xm10, xm11, q2020
    movd           xm9, [dst4q+stride3q +1]
    vinserti128     m6, xm10, 1
    pinsrd         xm9, [dst4q+strideq*4+1], 1
    psrldq        xm11, 2
    pmovzxbw       m10, [leftq-1]
    shufps        xm11, xm9, q1020
    movu            m9, [blend_4x8_0]
    vinserti128     m5, xm11, 1
    vpblendvb       m6, m10, m9
 %endif
%else
    lea            r13, [blend_8x8_0+8]
    movq           xm6, [top2q         -1]
    vbroadcasti128  m5, [dstq+strideq*1-1]
    vbroadcasti128  m9, [dstq+strideq*2-1]
    movhps         xm6, [dstq+strideq*0-1]
    movu           m11, [r13+hq*2*1+16*1]
    punpcklqdq     m10, m5, m9
    vinserti128     m5, [dstq+stride3q -1], 1
    vinserti128     m9, [dstq+strideq*4-1], 1
    vpblendd        m6, m10, 0xF0
    vpblendvb       m6, [rsp+gprsize+80+hq*8+64-8*1], m11
    psrldq          m5, 2
    psrldq          m9, 2
    punpcklqdq      m5, m9
%endif
    ret
.d5k0:
.d6k0:
.d7k0:
%if %1 == 4
 %if %2 == 4
    movd           xm6, [topq+strideq*1  ]
    vpbroadcastd    m5, [dstq+strideq*1  ]
    vpbroadcastd    m9, [dstq+strideq*2  ]
    vpblendd       xm6, [dstq+strideq*0-4], 0x2
    vpblendd        m5, m9, 0x22
    vpblendd        m6, m5, 0x30
    vinserti128     m5, [dstq+stride3q    ], 1
    vpblendd        m5, [dstq+strideq*4-20], 0x20
 %else
    movd           xm6, [topq +strideq*1]
    movd           xm5, [dstq +strideq*1]
    movd           xm9, [dstq +stride3q ]
    movd          xm10, [dst4q+strideq*1]
    movd          xm11, [dst4q+stride3q ]
    pinsrd         xm6, [dstq +strideq*0], 1
    pinsrd         xm5, [dstq +strideq*2], 1
    pinsrd         xm9, [dst4q+strideq*0], 1
    pinsrd        xm10, [dst4q+strideq*2], 1
    pinsrd        xm11, [dst4q+strideq*4], 1
    punpcklqdq     xm6, xm5
    punpcklqdq     xm5, xm9
    punpcklqdq     xm9, xm10
    punpcklqdq    xm10, xm11
    vinserti128     m6, xm9, 1
    vinserti128     m5, xm10, 1
 %endif
%else
    movq           xm6, [top2q         ]
    movq           xm5, [dstq+strideq*1]
    movq           xm9, [dstq+stride3q ]
    movhps         xm6, [dstq+strideq*0]
    movhps         xm5, [dstq+strideq*2]
    movhps         xm9, [dstq+strideq*4]
    vinserti128     m6, xm5, 1
    vinserti128     m5, xm9, 1
%endif
    ret
.d0k1:
%if %1 == 4
 %if %2 == 4
    movd           xm6, [dstq +strideq*2-2]
    movd           xm9, [dstq +stride3q -2]
    movd           xm5, [topq +strideq*0+2]
    movd          xm10, [topq +strideq*1+2]
    pinsrw         xm6, [leftq+4], 0
    pinsrw         xm9, [leftq+6], 0
    vinserti128     m5, [dstq +strideq*0+2], 1
    vinserti128    m10, [dstq +strideq*1+2], 1
    vinserti128     m6, [dst4q+strideq*0-2], 1
    vinserti128     m9, [dst4q+strideq*1-2], 1
    punpckldq       m5, m10
    punpckldq       m6, m9
 %else
    movq           xm6, [dstq +strideq*2-2]
    movd          xm10, [dst4q+strideq*2-2]
    movd           xm5, [topq +strideq*0+2]
    movq           xm9, [dst4q+strideq*0-2]
    movhps         xm6, [dstq +stride3q -2]
    pinsrw        xm10, [dst4q+stride3q   ], 3
    pinsrd         xm5, [topq +strideq*1+2], 1
    movhps         xm9, [dst4q+strideq*1-2]
    pinsrd        xm10, [dst8q+strideq*0-2], 2
    pinsrd         xm5, [dstq +strideq*0+2], 2
    pinsrd        xm10, [dst8q+strideq*1-2], 3
    pinsrd         xm5, [dstq +strideq*1+2], 3
    shufps        xm11, xm6, xm9, q3131
    shufps         xm6, xm9, q2020
    movu            m9, [blend_4x8_3+8]
    vinserti128     m6, xm10, 1
    vinserti128     m5, xm11, 1
    vpblendvb       m6, [rsp+gprsize+16+8], m9
 %endif
%else
    lea            r13, [blend_8x8_1+16]
    movq           xm6, [dstq +strideq*2-2]
    movq           xm9, [dstq +stride3q -2]
    movq           xm5, [top1q          +2]
    movq          xm10, [top2q          +2]
    movu           m11, [r13+hq*2*2+16*2]
    vinserti128     m6, [dst4q+strideq*0-2], 1
    vinserti128     m9, [dst4q+strideq*1-2], 1
    vinserti128     m5, [dstq +strideq*0+2], 1
    vinserti128    m10, [dstq +strideq*1+2], 1
    punpcklqdq      m6, m9
    punpcklqdq      m5, m10
    vpblendvb       m6, [rsp+gprsize+16+hq*8+64+8*2], m11
%endif
    ret
.d1k1:
%if %1 == 4
 %if %2 == 4
    vpbroadcastq    m6, [dstq+strideq*1-2]
    vpbroadcastq    m9, [dstq+strideq*2-2]
    movd           xm5, [topq+strideq*1+2]
    movd          xm10, [dstq+strideq*0+2]
    psrldq         m11, m6, 4
    psrldq         m12, m9, 4
    vpblendd        m5, m11, 0x10
    movq          xm11, [leftq+2]
    vinserti128     m6, [dstq+stride3q -2], 1
    punpckldq     xm11, xm11
    vpblendd       m10, m12, 0x10
    pcmpeqd        m12, m12
    pmovzxwd       m11, xm11
    psrld          m12, 16
    punpckldq       m6, m9
    vpbroadcastd    m9, [dstq+strideq*4-2]
    vpblendvb       m6, m11, m12
    punpckldq       m5, m10
    vpblendd        m6, m9, 0x20
 %else
    movd           xm5, [topq +strideq*1+2]
    movq           xm6, [dstq +strideq*1-2]
    movq           xm9, [dstq +stride3q -2]
    movq          xm10, [dst4q+strideq*1-2]
    movd          xm11, [dst4q+stride3q -2]
    pinsrd         xm5, [dstq +strideq*0+2], 1
    movhps         xm6, [dstq +strideq*2-2]
    movhps         xm9, [dst4q+strideq*0-2]
    movhps        xm10, [dst4q+strideq*2-2]
    pinsrd        xm11, [dst4q+strideq*4-2], 1
    shufps         xm5, xm6, q3110
    shufps         xm6, xm9, q2020
    shufps         xm9, xm10, q3131
    shufps        xm10, xm11, q1020
    movu           m11, [blend_4x8_2+4]
    vinserti128     m6, xm10, 1
    vinserti128     m5, xm9, 1
    vpblendvb       m6, [rsp+gprsize+16+4], m11
 %endif
%else
    lea            r13, [blend_8x8_1+16]
    movq           xm5, [top2q         +2]
    vbroadcasti128  m6, [dstq+strideq*1-2]
    vbroadcasti128  m9, [dstq+strideq*2-2]
    movhps         xm5, [dstq+strideq*0+2]
    shufps         m10, m6, m9, q2121
    vinserti128     m6, [dstq+stride3q -2], 1
    vinserti128     m9, [dstq+strideq*4-2], 1
    movu           m11, [r13+hq*2*1+16*1]
    vpblendd        m5, m10, 0xF0
    punpcklqdq      m6, m9
    vpblendvb       m6, [rsp+gprsize+16+hq*8+64+8*1], m11
%endif
    ret
.d2k1:
%if %1 == 4
 %if %2 == 4
    movq          xm11, [leftq]
    movq           xm6, [dstq+strideq*0-2]
    movq           xm9, [dstq+strideq*1-2]
    vinserti128     m6, [dstq+strideq*2-2], 1
    vinserti128     m9, [dstq+stride3q -2], 1
    punpckldq     xm11, xm11
    psrldq          m5, m6, 4
    psrldq         m10, m9, 4
    pmovzxwd       m11, xm11
    punpckldq       m6, m9
    punpckldq       m5, m10
    pblendw         m6, m11, 0x05
 %else
    movq           xm5, [dstq +strideq*0-2]
    movq           xm9, [dstq +strideq*2-2]
    movq          xm10, [dst4q+strideq*0-2]
    movq          xm11, [dst4q+strideq*2-2]
    movhps         xm5, [dstq +strideq*1-2]
    movhps         xm9, [dstq +stride3q -2]
    movhps        xm10, [dst4q+strideq*1-2]
    movhps        xm11, [dst4q+stride3q -2]
    shufps         xm6, xm5, xm9, q2020
    shufps         xm5, xm9, q3131
    shufps         xm9, xm10, xm11, q2020
    shufps        xm10, xm11, q3131
    pmovzxwd       m11, [leftq]
    vinserti128     m6, xm9, 1
    vinserti128     m5, xm10, 1
    pblendw         m6, m11, 0x55
 %endif
%else
    mova           m11, [rsp+gprsize+16+hq*8+64]
    movu           xm5, [dstq+strideq*0-2]
    movu           xm9, [dstq+strideq*1-2]
    vinserti128     m5, [dstq+strideq*2-2], 1
    vinserti128     m9, [dstq+stride3q -2], 1
    shufps          m6, m5, m9, q1010
    shufps          m5, m9, q2121
    pblendw         m6, m11, 0x11
%endif
    ret
.d3k1:
%if %1 == 4
 %if %2 == 4
    vpbroadcastq   m11, [dstq+strideq*1-2]
    vpbroadcastq   m12, [dstq+strideq*2-2]
    movd           xm6, [topq+strideq*1-2]
    movd           xm9, [dstq+strideq*0-2]
    pblendw        m11, [leftq-16+2], 0x01
    pblendw        m12, [leftq-16+4], 0x01
    pinsrw         xm9, [leftq- 0+0], 0
    psrldq          m5, m11, 4
    psrldq         m10, m12, 4
    vinserti128     m5, [dstq+stride3q +2], 1
    vinserti128    m10, [dstq+strideq*4+2], 1
    vpblendd        m6, m11, 0x10
    vpblendd        m9, m12, 0x10
    punpckldq       m6, m9
    punpckldq       m5, m10
 %else
    movd           xm6, [topq +strideq*1-2]
    movq           xm5, [dstq +strideq*1-2]
    movq           xm9, [dstq +stride3q -2]
    movq          xm10, [dst4q+strideq*1-2]
    movd          xm11, [dst4q+stride3q +2]
    pinsrw         xm6, [dstq +strideq*0  ], 3
    movhps         xm5, [dstq +strideq*2-2]
    movhps         xm9, [dst4q+strideq*0-2]
    movhps        xm10, [dst4q+strideq*2-2]
    pinsrd        xm11, [dst4q+strideq*4+2], 1
    shufps         xm6, xm5, q2010
    shufps         xm5, xm9, q3131
    shufps         xm9, xm10, q2020
    shufps        xm10, xm11, q1031
    movu           m11, [blend_4x8_2]
    vinserti128     m6, xm9, 1
    vinserti128     m5, xm10, 1
    vpblendvb       m6, [rsp+gprsize+16-4], m11
 %endif
%else
    lea            r13, [blend_8x8_1+8]
    movq           xm6, [top2q         -2]
    vbroadcasti128  m5, [dstq+strideq*1-2]
    vbroadcasti128 m10, [dstq+strideq*2-2]
    movhps         xm6, [dstq+strideq*0-2]
    punpcklqdq      m9, m5, m10
    vinserti128     m5, [dstq+stride3q -2], 1
    vinserti128    m10, [dstq+strideq*4-2], 1
    movu           m11, [r13+hq*2*1+16*1]
    vpblendd        m6, m9, 0xF0
    shufps          m5, m10, q2121
    vpblendvb       m6, [rsp+gprsize+16+hq*8+64-8*1], m11
%endif
    ret
.d4k1:
%if %1 == 4
 %if %2 == 4
    vinserti128     m6, [dstq +strideq*0-2], 1
    vinserti128     m9, [dstq +strideq*1-2], 1
    movd           xm5, [dstq +strideq*2+2]
    movd          xm10, [dstq +stride3q +2]
    pblendw         m6, [leftq-16+0], 0x01
    pblendw         m9, [leftq-16+2], 0x01
    vinserti128     m5, [dst4q+strideq*0+2], 1
    vinserti128    m10, [dst4q+strideq*1+2], 1
    vpblendd        m6, [topq +strideq*0-2], 0x01
    vpblendd        m9, [topq +strideq*1-2], 0x01
    punpckldq       m5, m10
    punpckldq       m6, m9
 %else
    movd           xm6, [topq +strideq*0-2]
    movq           xm5, [dstq +strideq*2-2]
    movq           xm9, [dst4q+strideq*0-2]
    movd          xm10, [dst4q+strideq*2+2]
    pinsrd         xm6, [topq +strideq*1-2], 1
    movhps         xm5, [dstq +stride3q -2]
    movhps         xm9, [dst4q+strideq*1-2]
    pinsrd        xm10, [dst4q+stride3q +2], 1
    pinsrd         xm6, [dstq +strideq*0-2], 2
    pinsrd        xm10, [dst8q+strideq*0+2], 2
    pinsrd         xm6, [dstq +strideq*1-2], 3
    pinsrd        xm10, [dst8q+strideq*1+2], 3
    shufps        xm11, xm5, xm9, q2020
    shufps         xm5, xm9, q3131
    movu            m9, [blend_4x8_3]
    vinserti128     m6, xm11, 1
    vinserti128     m5, xm10, 1
    vpblendvb       m6, [rsp+gprsize+16-8], m9
 %endif
%else
    lea            r13, [blend_8x8_1]
    movu           m11, [r13+hq*2*2+16*2]
    movq           xm6, [top1q          -2]
    movq           xm9, [top2q          -2]
    movq           xm5, [dstq +strideq*2+2]
    movq          xm10, [dstq +stride3q +2]
    vinserti128     m6, [dstq +strideq*0-2], 1
    vinserti128     m9, [dstq +strideq*1-2], 1
    vinserti128     m5, [dst4q+strideq*0+2], 1
    vinserti128    m10, [dst4q+strideq*1+2], 1
    punpcklqdq      m6, m9
    vpblendvb       m6, [rsp+gprsize+16+hq*8+64-8*2], m11
    punpcklqdq      m5, m10
%endif
    ret
.d5k1:
%if %1 == 4
 %if %2 == 4
    movd           xm6, [topq +strideq*0-1]
    movd           xm9, [topq +strideq*1-1]
    movd           xm5, [dstq +strideq*2+1]
    movd          xm10, [dstq +stride3q +1]
    pcmpeqd        m12, m12
    pmovzxbw       m11, [leftq-8+1]
    psrld          m12, 24
    vinserti128     m6, [dstq +strideq*0-1], 1
    vinserti128     m9, [dstq +strideq*1-1], 1
    vinserti128     m5, [dst4q+strideq*0+1], 1
    vinserti128    m10, [dst4q+strideq*1+1], 1
    punpckldq       m6, m9
    pxor            m9, m9
    vpblendd       m12, m9, 0x0F
    punpckldq       m5, m10
    vpblendvb       m6, m11, m12
 %else
    movd           xm6, [topq +strideq*0-1]
    movq           xm5, [dstq +strideq*2-1]
    movq           xm9, [dst4q+strideq*0-1]
    movd          xm10, [dst4q+strideq*2+1]
    pinsrd         xm6, [topq +strideq*1-1], 1
    movhps         xm5, [dstq +stride3q -1]
    movhps         xm9, [dst4q+strideq*1-1]
    pinsrd        xm10, [dst4q+stride3q +1], 1
    pinsrd         xm6, [dstq +strideq*0-1], 2
    pinsrd        xm10, [dst8q+strideq*0+1], 2
    pinsrd         xm6, [dstq +strideq*1-1], 3
    pinsrd        xm10, [dst8q+strideq*1+1], 3
    shufps        xm11, xm5, xm9, q2020
    vinserti128     m6, xm11, 1
    pmovzxbw       m11, [leftq-3]
    psrldq         xm5, 2
    psrldq         xm9, 2
    shufps         xm5, xm9, q2020
    movu            m9, [blend_4x8_1]
    vinserti128     m5, xm10, 1
    vpblendvb       m6, m11, m9
 %endif
%else
    lea            r13, [blend_8x8_0]
    movu           m11, [r13+hq*2*2+16*2]
    movq           xm6, [top1q          -1]
    movq           xm9, [top2q          -1]
    movq           xm5, [dstq +strideq*2+1]
    movq          xm10, [dstq +stride3q +1]
    vinserti128     m6, [dstq +strideq*0-1], 1
    vinserti128     m9, [dstq +strideq*1-1], 1
    vinserti128     m5, [dst4q+strideq*0+1], 1
    vinserti128    m10, [dst4q+strideq*1+1], 1
    punpcklqdq      m6, m9
    punpcklqdq      m5, m10
    vpblendvb       m6, [rsp+gprsize+80+hq*8+64-8*2], m11
%endif
    ret
.d6k1:
%if %1 == 4
 %if %2 == 4
    movd           xm6, [topq +strideq*0]
    movd           xm9, [topq +strideq*1]
    movd           xm5, [dstq +strideq*2]
    movd          xm10, [dstq +stride3q ]
    vinserti128     m6, [dstq +strideq*0], 1
    vinserti128     m9, [dstq +strideq*1], 1
    vinserti128     m5, [dst4q+strideq*0], 1
    vinserti128    m10, [dst4q+strideq*1], 1
    punpckldq       m6, m9
    punpckldq       m5, m10
 %else
    movd           xm5, [dstq +strideq*2]
    movd           xm6, [topq +strideq*0]
    movd           xm9, [dst4q+strideq*2]
    pinsrd         xm5, [dstq +stride3q ], 1
    pinsrd         xm6, [topq +strideq*1], 1
    pinsrd         xm9, [dst4q+stride3q ], 1
    pinsrd         xm5, [dst4q+strideq*0], 2
    pinsrd         xm6, [dstq +strideq*0], 2
    pinsrd         xm9, [dst8q+strideq*0], 2
    pinsrd         xm5, [dst4q+strideq*1], 3
    pinsrd         xm6, [dstq +strideq*1], 3
    pinsrd         xm9, [dst8q+strideq*1], 3
    vinserti128     m6, xm5, 1
    vinserti128     m5, xm9, 1
 %endif
%else
    movq           xm5, [dstq +strideq*2]
    movq           xm9, [dst4q+strideq*0]
    movq           xm6, [top1q          ]
    movq          xm10, [dstq +strideq*0]
    movhps         xm5, [dstq +stride3q ]
    movhps         xm9, [dst4q+strideq*1]
    movhps         xm6, [top2q          ]
    movhps        xm10, [dstq +strideq*1]
    vinserti128     m5, xm9, 1
    vinserti128     m6, xm10, 1
%endif
    ret
.d7k1:
%if %1 == 4
 %if %2 == 4
    movd           xm5, [dstq +strideq*2-1]
    movd           xm9, [dstq +stride3q -1]
    movd           xm6, [topq +strideq*0+1]
    movd          xm10, [topq +strideq*1+1]
    pinsrb         xm5, [leftq+ 5], 0
    pinsrb         xm9, [leftq+ 7], 0
    vinserti128     m6, [dstq +strideq*0+1], 1
    vinserti128    m10, [dstq +strideq*1+1], 1
    vinserti128     m5, [dst4q+strideq*0-1], 1
    vinserti128     m9, [dst4q+strideq*1-1], 1
    punpckldq       m6, m10
    punpckldq       m5, m9
 %else
    movd           xm6, [topq +strideq*0+1]
    movq           xm9, [dstq +strideq*2-1]
    movq          xm10, [dst4q+strideq*0-1]
    movd          xm11, [dst4q+strideq*2-1]
    pinsrd         xm6, [topq +strideq*1+1], 1
    movhps         xm9, [dstq +stride3q -1]
    movhps        xm10, [dst4q+strideq*1-1]
    pinsrd        xm11, [dst4q+stride3q -1], 1
    pinsrd         xm6, [dstq +strideq*0+1], 2
    pinsrd        xm11, [dst8q+strideq*0-1], 2
    pinsrd         xm6, [dstq +strideq*1+1], 3
    pinsrd        xm11, [dst8q+strideq*1-1], 3
    shufps         xm5, xm9, xm10, q2020
    vinserti128     m5, xm11, 1
    pmovzxbw       m11, [leftq+5]
    psrldq         xm9, 2
    psrldq        xm10, 2
    shufps         xm9, xm10, q2020
    movu           m10, [blend_4x8_1+8]
    vinserti128     m6, xm9, 1
    vpblendvb       m5, m11, m10
 %endif
%else
    lea            r13, [blend_8x8_0+16]
    movq           xm5, [dstq +strideq*2-1]
    movq           xm9, [dst4q+strideq*0-1]
    movq           xm6, [top1q          +1]
    movq          xm10, [dstq +strideq*0+1]
    movhps         xm5, [dstq +stride3q -1]
    movhps         xm9, [dst4q+strideq*1-1]
    movhps         xm6, [top2q          +1]
    movhps        xm10, [dstq +strideq*1+1]
    movu           m11, [r13+hq*2*2+16*2]
    vinserti128     m5, xm9, 1
    vinserti128     m6, xm10, 1
    vpblendvb       m5, [rsp+gprsize+80+hq*8+64+8*2], m11
%endif
    ret

.border_block:
 DEFINE_ARGS dst, stride, left, top, pri, sec, stride3, dst4, edge
%define rstk rsp
%assign stack_offset stack_offset_entry
%if %1 == 4 && %2 == 8
    PUSH            r9
 %assign regs_used 10
%else
 %assign regs_used 9
%endif
%if STACK_ALIGNMENT < 32
    PUSH  r%+regs_used
 %assign regs_used regs_used+1
%endif
    ALLOC_STACK 2*16+(%2+4)*32, 16
%define px rsp+2*16+2*32

    pcmpeqw        m14, m14
    psllw          m14, 15                  ; 0x8000

    ; prepare pixel buffers - body/right
%if %1 == 4
    INIT_XMM avx2
%endif
%if %2 == 8
    lea          dst4q, [dstq+strideq*4]
%endif
    lea       stride3q, [strideq*3]
    test         edgeb, 2                   ; have_right
    jz .no_right
    pmovzxbw        m1, [dstq+strideq*0]
    pmovzxbw        m2, [dstq+strideq*1]
    pmovzxbw        m3, [dstq+strideq*2]
    pmovzxbw        m4, [dstq+stride3q]
    mova     [px+0*32], m1
    mova     [px+1*32], m2
    mova     [px+2*32], m3
    mova     [px+3*32], m4
%if %2 == 8
    pmovzxbw        m1, [dst4q+strideq*0]
    pmovzxbw        m2, [dst4q+strideq*1]
    pmovzxbw        m3, [dst4q+strideq*2]
    pmovzxbw        m4, [dst4q+stride3q]
    mova     [px+4*32], m1
    mova     [px+5*32], m2
    mova     [px+6*32], m3
    mova     [px+7*32], m4
%endif
    jmp .body_done
.no_right:
%if %1 == 4
    movd           xm1, [dstq+strideq*0]
    movd           xm2, [dstq+strideq*1]
    movd           xm3, [dstq+strideq*2]
    movd           xm4, [dstq+stride3q]
    pmovzxbw       xm1, xm1
    pmovzxbw       xm2, xm2
    pmovzxbw       xm3, xm3
    pmovzxbw       xm4, xm4
    movq     [px+0*32], xm1
    movq     [px+1*32], xm2
    movq     [px+2*32], xm3
    movq     [px+3*32], xm4
%else
    pmovzxbw       xm1, [dstq+strideq*0]
    pmovzxbw       xm2, [dstq+strideq*1]
    pmovzxbw       xm3, [dstq+strideq*2]
    pmovzxbw       xm4, [dstq+stride3q]
    mova     [px+0*32], xm1
    mova     [px+1*32], xm2
    mova     [px+2*32], xm3
    mova     [px+3*32], xm4
%endif
    movd [px+0*32+%1*2], xm14
    movd [px+1*32+%1*2], xm14
    movd [px+2*32+%1*2], xm14
    movd [px+3*32+%1*2], xm14
%if %2 == 8
 %if %1 == 4
    movd           xm1, [dst4q+strideq*0]
    movd           xm2, [dst4q+strideq*1]
    movd           xm3, [dst4q+strideq*2]
    movd           xm4, [dst4q+stride3q]
    pmovzxbw       xm1, xm1
    pmovzxbw       xm2, xm2
    pmovzxbw       xm3, xm3
    pmovzxbw       xm4, xm4
    movq     [px+4*32], xm1
    movq     [px+5*32], xm2
    movq     [px+6*32], xm3
    movq     [px+7*32], xm4
 %else
    pmovzxbw       xm1, [dst4q+strideq*0]
    pmovzxbw       xm2, [dst4q+strideq*1]
    pmovzxbw       xm3, [dst4q+strideq*2]
    pmovzxbw       xm4, [dst4q+stride3q]
    mova     [px+4*32], xm1
    mova     [px+5*32], xm2
    mova     [px+6*32], xm3
    mova     [px+7*32], xm4
 %endif
    movd [px+4*32+%1*2], xm14
    movd [px+5*32+%1*2], xm14
    movd [px+6*32+%1*2], xm14
    movd [px+7*32+%1*2], xm14
%endif
.body_done:

    ; top
    test         edgeb, 4                    ; have_top
    jz .no_top
    test         edgeb, 1                    ; have_left
    jz .top_no_left
    test         edgeb, 2                    ; have_right
    jz .top_no_right
    pmovzxbw        m1, [topq+strideq*0-(%1/2)]
    pmovzxbw        m2, [topq+strideq*1-(%1/2)]
    movu  [px-2*32-%1], m1
    movu  [px-1*32-%1], m2
    jmp .top_done
.top_no_right:
    pmovzxbw        m1, [topq+strideq*0-%1]
    pmovzxbw        m2, [topq+strideq*1-%1]
    movu [px-2*32-%1*2], m1
    movu [px-1*32-%1*2], m2
    movd [px-2*32+%1*2], xm14
    movd [px-1*32+%1*2], xm14
    jmp .top_done
.top_no_left:
    test         edgeb, 2                   ; have_right
    jz .top_no_left_right
    pmovzxbw        m1, [topq+strideq*0]
    pmovzxbw        m2, [topq+strideq*1]
    mova   [px-2*32+0], m1
    mova   [px-1*32+0], m2
    movd   [px-2*32-4], xm14
    movd   [px-1*32-4], xm14
    jmp .top_done
.top_no_left_right:
%if %1 == 4
    movd           xm1, [topq+strideq*0]
    pinsrd         xm1, [topq+strideq*1], 1
    pmovzxbw       xm1, xm1
    movq   [px-2*32+0], xm1
    movhps [px-1*32+0], xm1
%else
    pmovzxbw       xm1, [topq+strideq*0]
    pmovzxbw       xm2, [topq+strideq*1]
    mova   [px-2*32+0], xm1
    mova   [px-1*32+0], xm2
%endif
    movd   [px-2*32-4], xm14
    movd   [px-1*32-4], xm14
    movd [px-2*32+%1*2], xm14
    movd [px-1*32+%1*2], xm14
    jmp .top_done
.no_top:
    movu   [px-2*32-%1], m14
    movu   [px-1*32-%1], m14
.top_done:

    ; left
    test         edgeb, 1                   ; have_left
    jz .no_left
    pmovzxbw       xm1, [leftq+ 0]
%if %2 == 8
    pmovzxbw       xm2, [leftq+ 8]
%endif
    movd   [px+0*32-4], xm1
    pextrd [px+1*32-4], xm1, 1
    pextrd [px+2*32-4], xm1, 2
    pextrd [px+3*32-4], xm1, 3
%if %2 == 8
    movd   [px+4*32-4], xm2
    pextrd [px+5*32-4], xm2, 1
    pextrd [px+6*32-4], xm2, 2
    pextrd [px+7*32-4], xm2, 3
%endif
    jmp .left_done
.no_left:
    movd   [px+0*32-4], xm14
    movd   [px+1*32-4], xm14
    movd   [px+2*32-4], xm14
    movd   [px+3*32-4], xm14
%if %2 == 8
    movd   [px+4*32-4], xm14
    movd   [px+5*32-4], xm14
    movd   [px+6*32-4], xm14
    movd   [px+7*32-4], xm14
%endif
.left_done:

    ; bottom
    DEFINE_ARGS dst, stride, dst8, dummy1, pri, sec, stride3, dummy3, edge
    test         edgeb, 8                   ; have_bottom
    jz .no_bottom
    lea          dst8q, [dstq+%2*strideq]
    test         edgeb, 1                   ; have_left
    jz .bottom_no_left
    test         edgeb, 2                   ; have_right
    jz .bottom_no_right
    pmovzxbw        m1, [dst8q-(%1/2)]
    pmovzxbw        m2, [dst8q+strideq-(%1/2)]
    movu   [px+(%2+0)*32-%1], m1
    movu   [px+(%2+1)*32-%1], m2
    jmp .bottom_done
.bottom_no_right:
    pmovzxbw        m1, [dst8q-%1]
    pmovzxbw        m2, [dst8q+strideq-%1]
    movu  [px+(%2+0)*32-%1*2], m1
    movu  [px+(%2+1)*32-%1*2], m2
%if %1 == 8
    movd  [px+(%2-1)*32+%1*2], xm14                ; overwritten by previous movu
%endif
    movd  [px+(%2+0)*32+%1*2], xm14
    movd  [px+(%2+1)*32+%1*2], xm14
    jmp .bottom_done
.bottom_no_left:
    test          edgeb, 2                  ; have_right
    jz .bottom_no_left_right
    pmovzxbw        m1, [dst8q]
    pmovzxbw        m2, [dst8q+strideq]
    mova   [px+(%2+0)*32+0], m1
    mova   [px+(%2+1)*32+0], m2
    movd   [px+(%2+0)*32-4], xm14
    movd   [px+(%2+1)*32-4], xm14
    jmp .bottom_done
.bottom_no_left_right:
%if %1 == 4
    movd           xm1, [dst8q]
    pinsrd         xm1, [dst8q+strideq], 1
    pmovzxbw       xm1, xm1
    movq   [px+(%2+0)*32+0], xm1
    movhps [px+(%2+1)*32+0], xm1
%else
    pmovzxbw       xm1, [dst8q]
    pmovzxbw       xm2, [dst8q+strideq]
    mova   [px+(%2+0)*32+0], xm1
    mova   [px+(%2+1)*32+0], xm2
%endif
    movd   [px+(%2+0)*32-4], xm14
    movd   [px+(%2+1)*32-4], xm14
    movd  [px+(%2+0)*32+%1*2], xm14
    movd  [px+(%2+1)*32+%1*2], xm14
    jmp .bottom_done
.no_bottom:
    movu   [px+(%2+0)*32-%1], m14
    movu   [px+(%2+1)*32-%1], m14
.bottom_done:

    ; actual filter
    INIT_YMM avx2
    DEFINE_ARGS dst, stride, pridmp, damping, pri, secdmp, stride3, zero
%undef edged
    ; register to shuffle values into after packing
    vbroadcasti128 m12, [shufb_lohi]

    mov       dampingd, r7m
    xor          zerod, zerod
    movifnidn     prid, prim
    sub       dampingd, 31
    movifnidn  secdmpd, secdmpm
    or            prid, 0
    jz .border_sec_only
    movd           xm0, prid
    lzcnt      pridmpd, prid
    add        pridmpd, dampingd
    cmovs      pridmpd, zerod
    mov        [rsp+0], pridmpq                 ; pri_shift
    or         secdmpd, 0
    jz .border_pri_only
    movd           xm1, secdmpd
    lzcnt      secdmpd, secdmpd
    add        secdmpd, dampingd
    cmovs      secdmpd, zerod
    mov        [rsp+8], secdmpq                 ; sec_shift

    DEFINE_ARGS dst, stride, pridmp, table, pri, secdmp, stride3
    lea         tableq, [tap_table]
    vpbroadcastb   m13, [tableq+pridmpq]        ; pri_shift_mask
    vpbroadcastb   m14, [tableq+secdmpq]        ; sec_shift_mask

    ; pri/sec_taps[k] [4 total]
    DEFINE_ARGS dst, stride, dir, table, pri, sec, stride3
    vpbroadcastb    m0, xm0                     ; pri_strength
    vpbroadcastb    m1, xm1                     ; sec_strength
    and           prid, 1
    lea           priq, [tableq+priq*2+8]       ; pri_taps
    lea           secq, [tableq+12]             ; sec_taps

    BORDER_PREP_REGS %1, %2
%if %1*%2*2/mmsize > 1
.border_v_loop:
%endif
    BORDER_LOAD_BLOCK %1, %2, 1
.border_k_loop:
    vpbroadcastb    m2, [priq+kq]               ; pri_taps
    vpbroadcastb    m3, [secq+kq]               ; sec_taps
    ACCUMULATE_TAP_WORD 0*2, [rsp+0], m13, m0, m2, %1, 1
    ACCUMULATE_TAP_WORD 2*2, [rsp+8], m14, m1, m3, %1, 1
    ACCUMULATE_TAP_WORD 6*2, [rsp+8], m14, m1, m3, %1, 1
    dec             kq
    jge .border_k_loop

    vpbroadcastd   m10, [pw_2048]
    BORDER_ADJUST_PIXEL %1, m10, 1
%if %1*%2*2/mmsize > 1
 %define vloop_lines (mmsize/(%1*2))
    lea           dstq, [dstq+strideq*vloop_lines]
    add           stkq, 32*vloop_lines
    dec             hd
    jg .border_v_loop
%endif
    RET

.border_pri_only:
 DEFINE_ARGS dst, stride, pridmp, table, pri, _, stride3
    lea         tableq, [tap_table]
    vpbroadcastb   m13, [tableq+pridmpq]        ; pri_shift_mask
 DEFINE_ARGS dst, stride, dir, table, pri, _, stride3
    vpbroadcastb    m0, xm0                     ; pri_strength
    and           prid, 1
    lea           priq, [tableq+priq*2+8]       ; pri_taps
    BORDER_PREP_REGS %1, %2
    vpbroadcastd    m1, [pw_2048]
%if %1*%2*2/mmsize > 1
.border_pri_v_loop:
%endif
    BORDER_LOAD_BLOCK %1, %2
.border_pri_k_loop:
    vpbroadcastb    m2, [priq+kq]               ; pri_taps
    ACCUMULATE_TAP_WORD 0*2, [rsp+0], m13, m0, m2, %1
    dec             kq
    jge .border_pri_k_loop
    BORDER_ADJUST_PIXEL %1, m1
%if %1*%2*2/mmsize > 1
 %define vloop_lines (mmsize/(%1*2))
    lea           dstq, [dstq+strideq*vloop_lines]
    add           stkq, 32*vloop_lines
    dec             hd
    jg .border_pri_v_loop
%endif
    RET

.border_sec_only:
 DEFINE_ARGS dst, stride, _, damping, _, secdmp, stride3, zero
    movd           xm1, secdmpd
    lzcnt      secdmpd, secdmpd
    add        secdmpd, dampingd
    cmovs      secdmpd, zerod
    mov        [rsp+8], secdmpq                 ; sec_shift
 DEFINE_ARGS dst, stride, _, table, _, secdmp, stride3
    lea         tableq, [tap_table]
    vpbroadcastb   m14, [tableq+secdmpq]        ; sec_shift_mask
 DEFINE_ARGS dst, stride, dir, table, _, sec, stride3
    vpbroadcastb    m1, xm1                     ; sec_strength
    lea           secq, [tableq+12]             ; sec_taps
    BORDER_PREP_REGS %1, %2
    vpbroadcastd    m0, [pw_2048]
%if %1*%2*2/mmsize > 1
.border_sec_v_loop:
%endif
    BORDER_LOAD_BLOCK %1, %2
.border_sec_k_loop:
    vpbroadcastb    m3, [secq+kq]               ; sec_taps
    ACCUMULATE_TAP_WORD 2*2, [rsp+8], m14, m1, m3, %1
    ACCUMULATE_TAP_WORD 6*2, [rsp+8], m14, m1, m3, %1
    dec             kq
    jge .border_sec_k_loop
    BORDER_ADJUST_PIXEL %1, m0
%if %1*%2*2/mmsize > 1
 %define vloop_lines (mmsize/(%1*2))
    lea           dstq, [dstq+strideq*vloop_lines]
    add           stkq, 32*vloop_lines
    dec             hd
    jg .border_sec_v_loop
%endif
    RET
%endmacro

CDEF_FILTER 8, 8
CDEF_FILTER 4, 8
CDEF_FILTER 4, 4

INIT_YMM avx2
cglobal cdef_dir, 3, 4, 15, src, stride, var, stride3
    lea       stride3q, [strideq*3]
    movq           xm0, [srcq+strideq*0]
    movq           xm1, [srcq+strideq*1]
    movq           xm2, [srcq+strideq*2]
    movq           xm3, [srcq+stride3q]
    lea           srcq, [srcq+strideq*4]
    vpbroadcastq    m4, [srcq+strideq*0]
    vpbroadcastq    m5, [srcq+strideq*1]
    vpbroadcastq    m6, [srcq+strideq*2]
    vpbroadcastq    m7, [srcq+stride3q]
    vpbroadcastd    m8, [pw_128]
    pxor            m9, m9

    vpblendd        m0, m0, m7, 0xf0
    vpblendd        m1, m1, m6, 0xf0
    vpblendd        m2, m2, m5, 0xf0
    vpblendd        m3, m3, m4, 0xf0

    punpcklbw       m0, m9
    punpcklbw       m1, m9
    punpcklbw       m2, m9
    punpcklbw       m3, m9

    psubw           m0, m8
    psubw           m1, m8
    psubw           m2, m8
    psubw           m3, m8

    ; shuffle registers to generate partial_sum_diag[0-1] together
    vpermq          m7, m0, q1032
    vpermq          m6, m1, q1032
    vpermq          m5, m2, q1032
    vpermq          m4, m3, q1032

    ; start with partial_sum_hv[0-1]
    paddw           m8, m0, m1
    paddw           m9, m2, m3
    phaddw         m10, m0, m1
    phaddw         m11, m2, m3
    paddw           m8, m9
    phaddw         m10, m11
    vextracti128   xm9, m8, 1
    vextracti128  xm11, m10, 1
    paddw          xm8, xm9                 ; partial_sum_hv[1]
    phaddw        xm10, xm11                ; partial_sum_hv[0]
    vinserti128     m8, xm10, 1
    vpbroadcastd    m9, [div_table+44]
    pmaddwd         m8, m8
    pmulld          m8, m9                  ; cost6[2a-d] | cost2[a-d]

    ; create aggregates [lower half]:
    ; m9 = m0:01234567+m1:x0123456+m2:xx012345+m3:xxx01234+
    ;      m4:xxxx0123+m5:xxxxx012+m6:xxxxxx01+m7:xxxxxxx0
    ; m10=             m1:7xxxxxxx+m2:67xxxxxx+m3:567xxxxx+
    ;      m4:4567xxxx+m5:34567xxx+m6:234567xx+m7:1234567x
    ; and [upper half]:
    ; m9 = m0:xxxxxxx0+m1:xxxxxx01+m2:xxxxx012+m3:xxxx0123+
    ;      m4:xxx01234+m5:xx012345+m6:x0123456+m7:01234567
    ; m10= m0:1234567x+m1:234567xx+m2:34567xxx+m3:4567xxxx+
    ;      m4:567xxxxx+m5:67xxxxxx+m6:7xxxxxxx
    ; and then shuffle m11 [shufw_6543210x], unpcklwd, pmaddwd, pmulld, paddd

    pslldq          m9, m1, 2
    psrldq         m10, m1, 14
    pslldq         m11, m2, 4
    psrldq         m12, m2, 12
    pslldq         m13, m3, 6
    psrldq         m14, m3, 10
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14
    pslldq         m11, m4, 8
    psrldq         m12, m4, 8
    pslldq         m13, m5, 10
    psrldq         m14, m5, 6
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14
    pslldq         m11, m6, 12
    psrldq         m12, m6, 4
    pslldq         m13, m7, 14
    psrldq         m14, m7, 2
    paddw           m9, m11
    paddw          m10, m12
    paddw           m9, m13
    paddw          m10, m14                 ; partial_sum_diag[0/1][8-14,zero]
    vbroadcasti128 m14, [shufw_6543210x]
    vbroadcasti128 m13, [div_table+16]
    vbroadcasti128 m12, [div_table+0]
    paddw           m9, m0                  ; partial_sum_diag[0/1][0-7]
    pshufb         m10, m14
    punpckhwd      m11, m9, m10
    punpcklwd       m9, m10
    pmaddwd        m11, m11
    pmaddwd         m9, m9
    pmulld         m11, m13
    pmulld          m9, m12
    paddd           m9, m11                 ; cost0[a-d] | cost4[a-d]

    ; merge horizontally and vertically for partial_sum_alt[0-3]
    paddw          m10, m0, m1
    paddw          m11, m2, m3
    paddw          m12, m4, m5
    paddw          m13, m6, m7
    phaddw          m0, m4
    phaddw          m1, m5
    phaddw          m2, m6
    phaddw          m3, m7

    ; create aggregates [lower half]:
    ; m4 = m10:01234567+m11:x0123456+m12:xx012345+m13:xxx01234
    ; m11=              m11:7xxxxxxx+m12:67xxxxxx+m13:567xxxxx
    ; and [upper half]:
    ; m4 = m10:xxx01234+m11:xx012345+m12:x0123456+m13:01234567
    ; m11= m10:567xxxxx+m11:67xxxxxx+m12:7xxxxxxx
    ; and then pshuflw m11 3012, unpcklwd, pmaddwd, pmulld, paddd

    pslldq          m4, m11, 2
    psrldq         m11, 14
    pslldq          m5, m12, 4
    psrldq         m12, 12
    pslldq          m6, m13, 6
    psrldq         m13, 10
    paddw           m4, m10
    paddw          m11, m12
    vpbroadcastd   m12, [div_table+44]
    paddw           m5, m6
    paddw          m11, m13                 ; partial_sum_alt[3/2] right
    vbroadcasti128 m13, [div_table+32]
    paddw           m4, m5                  ; partial_sum_alt[3/2] left
    pshuflw         m5, m11, q3012
    punpckhwd       m6, m11, m4
    punpcklwd       m4, m5
    pmaddwd         m6, m6
    pmaddwd         m4, m4
    pmulld          m6, m12
    pmulld          m4, m13
    paddd           m4, m6                  ; cost7[a-d] | cost5[a-d]

    ; create aggregates [lower half]:
    ; m5 = m0:01234567+m1:x0123456+m2:xx012345+m3:xxx01234
    ; m1 =             m1:7xxxxxxx+m2:67xxxxxx+m3:567xxxxx
    ; and [upper half]:
    ; m5 = m0:xxx01234+m1:xx012345+m2:x0123456+m3:01234567
    ; m1 = m0:567xxxxx+m1:67xxxxxx+m2:7xxxxxxx
    ; and then pshuflw m1 3012, unpcklwd, pmaddwd, pmulld, paddd

    pslldq          m5, m1, 2
    psrldq          m1, 14
    pslldq          m6, m2, 4
    psrldq          m2, 12
    pslldq          m7, m3, 6
    psrldq          m3, 10
    paddw           m5, m0
    paddw           m1, m2
    paddw           m6, m7
    paddw           m1, m3                  ; partial_sum_alt[0/1] right
    paddw           m5, m6                  ; partial_sum_alt[0/1] left
    pshuflw         m0, m1, q3012
    punpckhwd       m1, m5
    punpcklwd       m5, m0
    pmaddwd         m1, m1
    pmaddwd         m5, m5
    pmulld          m1, m12
    pmulld          m5, m13
    paddd           m5, m1                  ; cost1[a-d] | cost3[a-d]

    mova           xm0, [pd_47130256+ 16]
    mova            m1, [pd_47130256]
    phaddd          m9, m8
    phaddd          m5, m4
    phaddd          m9, m5
    vpermd          m0, m9                  ; cost[0-3]
    vpermd          m1, m9                  ; cost[4-7] | cost[0-3]

    ; now find the best cost
    pmaxsd         xm2, xm0, xm1
    pshufd         xm3, xm2, q1032
    pmaxsd         xm2, xm3
    pshufd         xm3, xm2, q2301
    pmaxsd         xm2, xm3 ; best cost

    ; find the idx using minpos
    ; make everything other than the best cost negative via subtraction
    ; find the min of unsigned 16-bit ints to sort out the negative values
    psubd          xm4, xm1, xm2
    psubd          xm3, xm0, xm2
    packssdw       xm3, xm4
    phminposuw     xm3, xm3

    ; convert idx to 32-bits
    psrld          xm3, 16
    movd           eax, xm3

    ; get idx^4 complement
    vpermd          m3, m1
    psubd          xm2, xm3
    psrld          xm2, 10
    movd        [varq], xm2
    RET

%if WIN64
DECLARE_REG_TMP 5, 6
%else
DECLARE_REG_TMP 8, 5
%endif

; lut:
; t0 t1 t2 t3 t4 t5 t6 t7
; T0 T1 T2 T3 T4 T5 T6 T7
; L0 L1 00 01 02 03 04 05
; L2 L3 10 11 12 13 14 15
; L4 L5 20 21 22 23 24 25
; L6 L7 30 31 32 33 34 35
; 4e 4f 40 41 42 43 44 45
; 5e 5f 50 51 52 53 54 55

%if HAVE_AVX512ICL

INIT_ZMM avx512icl
cglobal cdef_filter_4x4, 4, 8, 13, dst, stride, left, top, pri, sec, dir, damping, edge
%define base r7-edge_mask
    movq         xmm0, [dstq+strideq*0]
    movhps       xmm0, [dstq+strideq*1]
    lea            r7, [edge_mask]
    movq         xmm1, [topq+strideq*0-2]
    movhps       xmm1, [topq+strideq*1-2]
    mov           r6d, edgem
    vinserti32x4  ym0, ymm0, [leftq], 1
    lea            r2, [strideq*3]
    vinserti32x4  ym1, ymm1, [dstq+strideq*2], 1
    mova           m5, [base+lut_perm_4x4]
    vinserti32x4   m0, [dstq+r2], 2
    test          r6b, 0x08      ; avoid buffer overread
    jz .main
    lea            r3, [dstq+strideq*4-4]
    vinserti32x4   m1, [r3+strideq*0], 2
    vinserti32x4   m0, [r3+strideq*1], 3
.main:
    movifnidn    prid, prim
    mov           t0d, dirm
    mova           m3, [base+px_idx]
    mov           r3d, dampingm
    vpermi2b       m5, m0, m1    ; lut
    vpbroadcastd   m0, [base+pd_268435568] ; (1 << 28) + (7 << 4)
    pxor           m7, m7
    lea            r3, [r7+r3*8] ; gf_shr + (damping - 30) * 8
    vpermb         m6, m3, m5    ; px
    cmp           r6d, 0x0f
    jne .mask_edges              ; mask edges only if required
    test         prid, prid
    jz .sec_only
    vpaddd         m1, m3, [base+cdef_dirs+(t0+2)*4] {1to16} ; dir
    vpermb         m1, m1, m5    ; k0p0 k0p1 k1p0 k1p1
%macro CDEF_FILTER_4x4_PRI 0
    vpcmpub        k1, m6, m1, 6 ; px > pN
    psubb          m2, m1, m6
    lzcnt         r6d, prid
    vpsubb     m2{k1}, m6, m1    ; abs(diff)
    vpbroadcastb   m4, prid
    and          prid, 1
    vgf2p8affineqb m9, m2, [r3+r6*8] {1to8}, 0 ; abs(diff) >> shift
    movifnidn     t1d, secm
    vpbroadcastd  m10, [base+pri_tap+priq*4]
    vpsubb    m10{k1}, m7, m10   ; apply_sign(pri_tap)
    psubusb        m4, m9        ; imax(0, pri_strength - (abs(diff) >> shift)))
    pminub         m2, m4
    vpdpbusd       m0, m2, m10   ; sum
%endmacro
    CDEF_FILTER_4x4_PRI
    test          t1d, t1d       ; sec
    jz .end_no_clip
    call .sec
.end_clip:
    pminub         m4, m6, m1
    pmaxub         m1, m6
    pminub         m5, m2, m3
    pmaxub         m2, m3
    pminub         m4, m5
    pmaxub         m2, m1
    psrldq         m1, m4, 2
    psrldq         m3, m2, 2
    pminub         m1, m4
    vpcmpw         k1, m0, m7, 1
    vpshldd        m6, m0, 8
    pmaxub         m2, m3
    pslldq         m3, m1, 1
    psubw          m7, m0
    paddusw        m0, m6     ; clip >0xff
    vpsubusw   m0{k1}, m6, m7 ; clip <0x00
    pslldq         m4, m2, 1
    pminub         m1, m3
    pmaxub         m2, m4
    pmaxub         m0, m1
    pminub         m0, m2
    jmp .end
.sec_only:
    movifnidn     t1d, secm
    call .sec
.end_no_clip:
    vpshldd        m6, m0, 8  ; (px << 8) + ((sum > -8) << 4)
    paddw          m0, m6     ; (px << 8) + ((sum + (sum > -8) + 7) << 4)
.end:
    mova          xm1, [base+end_perm]
    vpermb         m0, m1, m0 ; output in bits 8-15 of each dword
    movd   [dstq+strideq*0], xm0
    pextrd [dstq+strideq*1], xm0, 1
    pextrd [dstq+strideq*2], xm0, 2
    pextrd [dstq+r2       ], xm0, 3
    RET
.mask_edges_sec_only:
    movifnidn     t1d, secm
    call .mask_edges_sec
    jmp .end_no_clip
ALIGN function_align
.mask_edges:
    vpbroadcastq   m8, [base+edge_mask+r6*8]
    test         prid, prid
    jz .mask_edges_sec_only
    vpaddd         m2, m3, [base+cdef_dirs+(t0+2)*4] {1to16}
    vpshufbitqmb   k1, m8, m2 ; index in-range
    mova           m1, m6
    vpermb     m1{k1}, m2, m5
    CDEF_FILTER_4x4_PRI
    test          t1d, t1d
    jz .end_no_clip
    call .mask_edges_sec
    jmp .end_clip
.mask_edges_sec:
    vpaddd         m4, m3, [base+cdef_dirs+(t0+4)*4] {1to16}
    vpaddd         m9, m3, [base+cdef_dirs+(t0+0)*4] {1to16}
    vpshufbitqmb   k1, m8, m4
    mova           m2, m6
    vpermb     m2{k1}, m4, m5
    vpshufbitqmb   k1, m8, m9
    mova           m3, m6
    vpermb     m3{k1}, m9, m5
    jmp .sec_main
ALIGN function_align
.sec:
    vpaddd         m2, m3, [base+cdef_dirs+(t0+4)*4] {1to16} ; dir + 2
    vpaddd         m3,     [base+cdef_dirs+(t0+0)*4] {1to16} ; dir - 2
    vpermb         m2, m2, m5 ; k0s0 k0s1 k1s0 k1s1
    vpermb         m3, m3, m5 ; k0s2 k0s3 k1s2 k1s3
.sec_main:
    vpbroadcastd   m8, [base+sec_tap]
    vpcmpub        k1, m6, m2, 6
    psubb          m4, m2, m6
    vpbroadcastb  m12, t1d
    lzcnt         t1d, t1d
    vpsubb     m4{k1}, m6, m2
    vpcmpub        k2, m6, m3, 6
    vpbroadcastq  m11, [r3+t1*8]
    gf2p8affineqb m10, m4, m11, 0
    psubb          m5, m3, m6
    mova           m9, m8
    vpsubb     m8{k1}, m7, m8
    psubusb       m10, m12, m10
    vpsubb     m5{k2}, m6, m3
    pminub         m4, m10
    vpdpbusd       m0, m4, m8
    gf2p8affineqb m11, m5, m11, 0
    vpsubb     m9{k2}, m7, m9
    psubusb       m12, m11
    pminub         m5, m12
    vpdpbusd       m0, m5, m9
    ret

DECLARE_REG_TMP 2, 7

;         lut top                lut bottom
; t0 t1 t2 t3 t4 t5 t6 t7  L4 L5 20 21 22 23 24 25
; T0 T1 T2 T3 T4 T5 T6 T7  L6 L7 30 31 32 33 34 35
; L0 L1 00 01 02 03 04 05  L8 L9 40 41 42 43 44 45
; L2 L3 10 11 12 13 14 15  La Lb 50 51 52 53 54 55
; L4 L5 20 21 22 23 24 25  Lc Ld 60 61 62 63 64 65
; L6 L7 30 31 32 33 34 35  Le Lf 70 71 72 73 74 75
; L8 L9 40 41 42 43 44 45  8e 8f 80 81 82 83 84 85
; La Lb 50 51 52 53 54 55  9e 9f 90 91 92 93 94 95

cglobal cdef_filter_4x8, 4, 9, 22, dst, stride, left, top, \
                                   pri, sec, dir, damping, edge
%define base r8-edge_mask
    vpbroadcastd ym21, strided
    mov           r6d, edgem
    lea            r8, [edge_mask]
    movq          xm1, [topq+strideq*0-2]
    pmulld       ym21, [base+pd_01234567]
    kxnorb         k1, k1, k1
    movq          xm2, [topq+strideq*1-2]
    vpgatherdq m0{k1}, [dstq+ym21]  ; +0+1 +2+3 +4+5 +6+7
    mova          m14, [base+lut_perm_4x8a]
    movu          m15, [base+lut_perm_4x8b]
    test          r6b, 0x08         ; avoid buffer overread
    jz .main
    lea            r7, [dstq+strideq*8-2]
    vinserti32x4  ym1, [r7+strideq*0], 1
    vinserti32x4  ym2, [r7+strideq*1], 1
.main:
    punpcklqdq    ym1, ym2
    vinserti32x4   m1, [leftq], 2   ; -2-1 +8+9 left ____
    movifnidn    prid, prim
    mov           t0d, dirm
    mova          m16, [base+px_idx]
    mov           r3d, dampingm
    vpermi2b      m14, m0, m1    ; lut top
    vpermi2b      m15, m0, m1    ; lut bottom
    vpbroadcastd   m0, [base+pd_268435568] ; (1 << 28) + (7 << 4)
    pxor          m20, m20
    lea            r3, [r8+r3*8] ; gf_shr + (damping - 30) * 8
    vpermb         m2, m16, m14  ; pxt
    vpermb         m3, m16, m15  ; pxb
    mova           m1, m0
    cmp           r6b, 0x0f
    jne .mask_edges              ; mask edges only if required
    test         prid, prid
    jz .sec_only
    vpaddd         m6, m16, [base+cdef_dirs+(t0+2)*4] {1to16} ; dir
    vpermb         m4, m6, m14   ; pNt k0p0 k0p1 k1p0 k1p1
    vpermb         m5, m6, m15   ; pNb
%macro CDEF_FILTER_4x8_PRI 0
    vpcmpub        k1, m2, m4, 6 ; pxt > pNt
    vpcmpub        k2, m3, m5, 6 ; pxb > pNb
    psubb          m6, m4, m2
    psubb          m7, m5, m3
    lzcnt         r6d, prid
    vpsubb     m6{k1}, m2, m4    ; abs(diff_top)
    vpsubb     m7{k2}, m3, m5    ; abs(diff_bottom)
    vpbroadcastb  m13, prid
    vpbroadcastq   m9, [r3+r6*8]
    and          prid, 1
    vpbroadcastd  m11, [base+pri_tap+priq*4]
    vgf2p8affineqb m8, m6, m9, 0 ; abs(dt) >> shift
    vgf2p8affineqb m9, m7, m9, 0 ; abs(db) >> shift
    mova          m10, m11
    movifnidn     t1d, secm
    vpsubb    m10{k1}, m20, m11  ; apply_sign(pri_tap_top)
    vpsubb    m11{k2}, m20, m11  ; apply_sign(pri_tap_bottom)
    psubusb       m12, m13, m8   ; imax(0, pri_strength - (abs(dt) >> shift)))
    psubusb       m13, m13, m9   ; imax(0, pri_strength - (abs(db) >> shift)))
    pminub         m6, m12
    pminub         m7, m13
    vpdpbusd       m0, m6, m10   ; sum top
    vpdpbusd       m1, m7, m11   ; sum bottom
%endmacro
    CDEF_FILTER_4x8_PRI
    test          t1d, t1d       ; sec
    jz .end_no_clip
    call .sec
.end_clip:
    pminub        m10, m4, m2
    pminub        m12, m6, m8
    pminub        m11, m5, m3
    pminub        m13, m7, m9
    pmaxub         m4, m2
    pmaxub         m6, m8
    pmaxub         m5, m3
    pmaxub         m7, m9
    pminub        m10, m12
    pminub        m11, m13
    pmaxub         m4, m6
    pmaxub         m5, m7
    mov           r2d, 0xAAAAAAAA
    kmovd          k1, r2d
    kxnorb         k2, k2, k2       ;   hw   lw
    vpshrdd       m12, m0, m1, 16   ;  m1lw m0hw
    vpshrdd        m6, m10, m11, 16 ; m11lw m10hw
    vpshrdd        m8, m4, m5, 16   ;  m5lw m4hw
    vpblendmw  m7{k1}, m10, m11     ; m11hw m10lw
    vpblendmw  m9{k1}, m4, m5       ;  m5hw m4lw
    vpblendmw  m4{k1}, m0, m12      ;  m1lw m0lw
    vpblendmw  m5{k1}, m12, m1      ;  m1hw m0hw
    vpshrdd        m2, m3, 16
    pminub         m6, m7
    pmaxub         m8, m9
    mova         ym14, [base+end_perm]
    vpcmpw         k1, m4, m20, 1
    vpshldw        m2, m5, 8
    pslldq         m7, m6, 1
    pslldq         m9, m8, 1
    psubw          m5, m20, m4
    paddusw        m0, m4, m2 ; clip >0xff
    pminub         m6, m7
    pmaxub         m8, m9
    psubusw    m0{k1}, m2, m5 ; clip <0x00
    pmaxub         m0, m6
    pminub         m0, m8
    vpermb         m0, m14, m0
    vpscatterdd [dstq+ym21]{k2}, ym0
    RET
.sec_only:
    movifnidn     t1d, secm
    call .sec
.end_no_clip:
    mova          ym4, [base+end_perm]
    kxnorb         k1, k1, k1
    vpshldd        m2, m0, 8  ; (px << 8) + ((sum > -8) << 4)
    vpshldd        m3, m1, 8
    paddw          m0, m2     ; (px << 8) + ((sum + (sum > -8) + 7) << 4)
    paddw          m1, m3
    pslld          m0, 16
    vpshrdd        m0, m1, 16
    vpermb         m0, m4, m0 ; output in bits 8-15 of each word
    vpscatterdd [dstq+ym21]{k1}, ym0
    RET
.mask_edges_sec_only:
    movifnidn     t1d, secm
    call .mask_edges_sec
    jmp .end_no_clip
ALIGN function_align
.mask_edges:
    mov           t1d, r6d
    or            r6d, 8 ; top 4x4 has bottom
    or            t1d, 4 ; bottom 4x4 has top
    vpbroadcastq  m17, [base+edge_mask+r6*8]
    vpbroadcastq  m18, [base+edge_mask+t1*8]
    test         prid, prid
    jz .mask_edges_sec_only
    vpaddd         m6, m16, [base+cdef_dirs+(t0+2)*4] {1to16}
    vpshufbitqmb   k1, m17, m6 ; index in-range
    vpshufbitqmb   k2, m18, m6
    mova           m4, m2
    mova           m5, m3
    vpermb     m4{k1}, m6, m14
    vpermb     m5{k2}, m6, m15
    CDEF_FILTER_4x8_PRI
    test          t1d, t1d
    jz .end_no_clip
    call .mask_edges_sec
    jmp .end_clip
.mask_edges_sec:
    vpaddd        m10, m16, [base+cdef_dirs+(t0+4)*4] {1to16}
    vpaddd        m11, m16, [base+cdef_dirs+(t0+0)*4] {1to16}
    vpshufbitqmb   k1, m17, m10
    vpshufbitqmb   k2, m18, m10
    vpshufbitqmb   k3, m17, m11
    vpshufbitqmb   k4, m18, m11
    mova           m6, m2
    mova           m7, m3
    mova           m8, m2
    mova           m9, m3
    vpermb     m6{k1}, m10, m14
    vpermb     m7{k2}, m10, m15
    vpermb     m8{k3}, m11, m14
    vpermb     m9{k4}, m11, m15
    jmp .sec_main
ALIGN function_align
.sec:
    vpaddd         m8, m16, [base+cdef_dirs+(t0+4)*4] {1to16} ; dir + 2
    vpaddd         m9, m16, [base+cdef_dirs+(t0+0)*4] {1to16} ; dir - 2
    vpermb         m6, m8, m14 ; pNt k0s0 k0s1 k1s0 k1s1
    vpermb         m7, m8, m15 ; pNb
    vpermb         m8, m9, m14 ; pNt k0s2 k0s3 k1s2 k1s3
    vpermb         m9, m9, m15 ; pNb
.sec_main:
    vpbroadcastb  m18, t1d
    lzcnt         t1d, t1d
    vpcmpub        k1, m2, m6, 6
    vpcmpub        k2, m3, m7, 6
    vpcmpub        k3, m2, m8, 6
    vpcmpub        k4, m3, m9, 6
    vpbroadcastq  m17, [r3+t1*8]
    psubb         m10, m6, m2
    psubb         m11, m7, m3
    psubb         m12, m8, m2
    psubb         m13, m9, m3
    vpsubb    m10{k1}, m2, m6      ; abs(dt0)
    vpsubb    m11{k2}, m3, m7      ; abs(db0)
    vpsubb    m12{k3}, m2, m8      ; abs(dt1)
    vpsubb    m13{k4}, m3, m9      ; abs(db1)
    vpbroadcastd  m19, [base+sec_tap]
    gf2p8affineqb m14, m10, m17, 0 ; abs(dt0) >> shift
    gf2p8affineqb m15, m11, m17, 0 ; abs(db0) >> shift
    gf2p8affineqb m16, m12, m17, 0 ; abs(dt1) >> shift
    gf2p8affineqb m17, m13, m17, 0 ; abs(db1) >> shift
    psubusb       m14, m18, m14    ; imax(0, sec_strength - (abs(dt0) >> shift)))
    psubusb       m15, m18, m15    ; imax(0, sec_strength - (abs(db0) >> shift)))
    psubusb       m16, m18, m16    ; imax(0, sec_strength - (abs(dt1) >> shift)))
    psubusb       m17, m18, m17    ; imax(0, sec_strength - (abs(db1) >> shift)))
    pminub        m10, m14
    pminub        m11, m15
    pminub        m12, m16
    pminub        m13, m17
    mova          m14, m19
    mova          m15, m19
    mova          m16, m19
    vpsubb    m14{k1}, m20, m19    ; apply_sign(sec_tap_top_0)
    vpsubb    m15{k2}, m20, m19    ; apply_sign(sec_tap_bottom_0)
    vpsubb    m16{k3}, m20, m19    ; apply_sign(sec_tap_top_1)
    vpsubb    m19{k4}, m20, m19    ; apply_sign(sec_tap_bottom_1)
    vpdpbusd       m0, m10, m14
    vpdpbusd       m1, m11, m15
    vpdpbusd       m0, m12, m16
    vpdpbusd       m1, m13, m19
    ret

;         lut tl                   lut tr
; t0 t1 t2 t3 t4 t5 t6 t7  t6 t7 t8 t9 ta tb tc td
; T0 T1 T2 T3 T4 T5 T6 T7  T6 T7 T8 T9 TA TB TC TD
; L0 L1 00 01 02 03 04 05  04 05 06 07 08 09 0a 0b
; L2 L3 10 11 12 13 14 15  14 15 16 17 18 19 1a 1b
; L4 L5 20 21 22 23 24 25  24 25 26 27 28 29 2a 2b
; L6 L7 30 31 32 33 34 35  34 35 36 37 38 39 3a 3b
; L8 L9 40 41 42 43 44 45  44 45 46 47 48 49 4a 4b
; La Lb 50 51 52 53 54 55  54 55 56 57 58 59 5a 5b
;         lut bl                   lut br
; L4 L5 20 21 22 23 24 25  24 25 26 27 28 29 2a 2b
; L6 L7 30 31 32 33 34 35  34 35 36 37 38 39 3a 3b
; L8 L9 40 41 42 43 44 45  44 45 46 47 48 49 4a 4b
; La Lb 50 51 52 53 54 55  54 55 56 57 58 59 5a 5b
; Lc Ld 60 61 62 63 64 65  64 65 66 67 68 69 6a 6b
; Le Lf 70 71 72 73 74 75  74 75 76 77 78 79 7a 7b
; 8e 8f 80 81 82 83 84 85  84 85 86 87 88 89 8a 8b
; 9e 9f 90 91 92 93 94 95  94 95 96 97 98 99 9a 9b

cglobal cdef_filter_8x8, 4, 11, 32, 4*64, dst, stride, left, top, \
                                          pri, sec, dir, damping, edge
%define base r8-edge_mask
    mov           r6d, edgem
    lea           r10, [dstq+strideq*4-2]
    movu         xmm0, [topq+strideq*0-2]
    movu         xmm1, [dstq+strideq*2-2]
    movu         xmm2, [r10 +strideq*2  ]
    lea            r8, [edge_mask]
    lea            r9, [strideq*3]
    pmovzxwq      m10, [leftq-4]
    vinserti32x4  ym0, ymm0, [topq+strideq*1-2], 1
    vinserti32x4  ym1, ymm1, [dstq+r9       -2], 1
    vinserti32x4  ym2, ymm2, [r10 +r9         ], 1
    lea            r7, [r10 +strideq*4  ]
    pmovzxwq      m11, [leftq+4]
    vinserti32x4   m0, [dstq+strideq*0-2], 2
    vinserti32x4   m1, [r10 +strideq*0  ], 2
    mova          m12, [base+lut_perm_8x8a]
    movu          m13, [base+lut_perm_8x8b]
    vinserti32x4   m0, [dstq+strideq*1-2], 3
    vinserti32x4   m1, [r10 +strideq*1  ], 3
    test          r6b, 0x08       ; avoid buffer overread
    jz .main
    vinserti32x4   m2, [r7  +strideq*0], 2
    vinserti32x4   m2, [r7  +strideq*1], 3
.main:
    mov           t1d, 0x11111100
    mova          m14, m12
    mova          m15, m13
    kmovd          k1, t1d
    kshiftrd       k2, k1, 8
    movifnidn    prid, prim
    mov           t0d, dirm
    mova          m30, [base+px_idx]
    mov           r3d, dampingm
    vpermi2b      m12, m0, m1     ; lut tl
    vpermi2b      m14, m1, m2     ; lut bl
    vpermi2b      m13, m0, m1     ; lut tr
    vpermi2b      m15, m1, m2     ; lut br
    vpblendmw m12{k1}, m12, m10
    vpblendmw m14{k2}, m14, m11
    vpbroadcastd   m0, [base+pd_268435568] ; (1 << 28) + (7 << 4)
    pxor          m31, m31
    lea            r3, [r8+r3*8]  ; gf_shr + (damping - 30) * 8
    vpermb         m4, m30, m12   ; pxtl
    vpermb         m5, m30, m13   ; pxtr
    vpermb         m6, m30, m14   ; pxbl
    vpermb         m7, m30, m15   ; pxbr
    mova           m1, m0
    mova           m2, m0
    mova           m3, m0
    cmp           r6b, 0x0f
    jne .mask_edges               ; mask edges only if required
    test         prid, prid
    jz .sec_only
    vpaddd        m11, m30, [base+cdef_dirs+(t0+2)*4] {1to16} ; dir
    vpermb         m8, m11, m12   ; pNtl k0p0 k0p1 k1p0 k1p1
    vpermb         m9, m11, m13   ; pNtr
    vpermb        m10, m11, m14   ; pNbl
    vpermb        m11, m11, m15   ; pNbr
%macro CDEF_FILTER_8x8_PRI 0
    vpcmpub        k1, m4, m8, 6  ; pxtl > pNtl
    vpcmpub        k2, m5, m9, 6  ; pxtr > pNtr
    vpcmpub        k3, m6, m10, 6 ; pxbl > pNbl
    vpcmpub        k4, m7, m11, 6 ; pxbr > pNbr
    psubb         m16, m8, m4
    psubb         m17, m9, m5
    psubb         m18, m10, m6
    psubb         m19, m11, m7
    lzcnt         r6d, prid
    vpsubb    m16{k1}, m4, m8     ; abs(diff_tl)
    vpsubb    m17{k2}, m5, m9     ; abs(diff_tr)
    vpsubb    m18{k3}, m6, m10    ; abs(diff_bl)
    vpsubb    m19{k4}, m7, m11    ; abs(diff_br)
    vpbroadcastq  m28, [r3+r6*8]
    vpbroadcastb  m29, prid
    and          prid, 1
    vpbroadcastd  m27, [base+pri_tap+priq*4]
    vgf2p8affineqb m20, m16, m28, 0 ; abs(dtl) >> shift
    vgf2p8affineqb m21, m17, m28, 0 ; abs(dtr) >> shift
    vgf2p8affineqb m22, m18, m28, 0 ; abs(dbl) >> shift
    vgf2p8affineqb m23, m19, m28, 0 ; abs(dbl) >> shift
    mova          m24, m27
    mova          m25, m27
    mova          m26, m27
    movifnidn     t1d, secm
    vpsubb    m24{k1}, m31, m27   ; apply_sign(pri_tap_tl)
    vpsubb    m25{k2}, m31, m27   ; apply_sign(pri_tap_tr)
    vpsubb    m26{k3}, m31, m27   ; apply_sign(pri_tap_tl)
    vpsubb    m27{k4}, m31, m27   ; apply_sign(pri_tap_tr)
    psubusb       m20, m29, m20   ; imax(0, pri_strength - (abs(dtl) >> shift)))
    psubusb       m21, m29, m21   ; imax(0, pri_strength - (abs(dtr) >> shift)))
    psubusb       m22, m29, m22   ; imax(0, pri_strength - (abs(dbl) >> shift)))
    psubusb       m23, m29, m23   ; imax(0, pri_strength - (abs(dbr) >> shift)))
    pminub        m16, m20
    pminub        m17, m21
    pminub        m18, m22
    pminub        m19, m23
    vpdpbusd       m0, m16, m24   ; sum tl
    vpdpbusd       m1, m17, m25   ; sum tr
    vpdpbusd       m2, m18, m26   ; sum bl
    vpdpbusd       m3, m19, m27   ; sum br
%endmacro
    CDEF_FILTER_8x8_PRI
    test          t1d, t1d        ; sec
    jz .end_no_clip
    call .sec
.end_clip:
    pminub        m20, m8, m4
    pminub        m24, m12, m16
    pminub        m21, m9, m5
    pminub        m25, m13, m17
    pminub        m22, m10, m6
    pminub        m26, m14, m18
    pminub        m23, m11, m7
    pminub        m27, m15, m19
    pmaxub         m8, m4
    pmaxub        m12, m16
    pmaxub         m9, m5
    pmaxub        m13, m17
    pmaxub        m10, m6
    pmaxub        m14, m18
    pmaxub        m11, m7
    pmaxub        m15, m19
    pminub        m20, m24
    pminub        m21, m25
    pminub        m22, m26
    pminub        m23, m27
    pmaxub         m8, m12
    pmaxub         m9, m13
    pmaxub        m10, m14
    pmaxub        m11, m15
    mov           r2d, 0xAAAAAAAA
    kmovd          k1, r2d
    vpshrdd       m24,  m0,  m1, 16
    vpshrdd       m25,  m2,  m3, 16
    vpshrdd       m12, m20, m21, 16
    vpshrdd       m14, m22, m23, 16
    vpshrdd       m16,  m8,  m9, 16
    vpshrdd       m18, m10, m11, 16
    vpblendmw m13{k1}, m20, m21
    vpblendmw m15{k1}, m22, m23
    vpblendmw m17{k1},  m8, m9
    vpblendmw m19{k1}, m10, m11
    vpblendmw m20{k1},  m0, m24
    vpblendmw m21{k1}, m24, m1
    vpblendmw m22{k1},  m2, m25
    vpblendmw m23{k1}, m25, m3
    vpshrdd        m4, m5, 16
    vpshrdd        m6, m7, 16
    pminub        m12, m13
    pminub        m14, m15
    pmaxub        m16, m17
    pmaxub        m18, m19
    mova           m8, [base+end_perm_w8clip]
    vpcmpw         k2, m20, m31, 1
    vpcmpw         k3, m22, m31, 1
    vpshldw        m4, m21, 8
    vpshldw        m6, m23, 8
    kunpckdq       k1, k1, k1
    kxnorb         k4, k4, k4
    vpshrdw       m11, m12, m14, 8
    vpshrdw       m15, m16, m18, 8
    vpblendmb m13{k1}, m12, m14
    vpblendmb m17{k1}, m16, m18
    psubw         m21, m31, m20
    psubw         m23, m31, m22
    paddusw        m0, m20, m4  ; clip >0xff
    paddusw        m1, m22, m6
    pminub        m11, m13
    pmaxub        m15, m17
    psubusw    m0{k2}, m4, m21  ; clip <0x00
    psubusw    m1{k3}, m6, m23
    psrlw          m0, 8
    vmovdqu8   m0{k1}, m1
    pmaxub         m0, m11
    pminub         m0, m15
    vpermb         m0, m8, m0
    add           r10, 2
    vextracti32x4 xm1, m0, 1
    vextracti32x4 xm2, m0, 2
    vextracti32x4 xm3, m0, 3
    movq   [dstq+strideq*0], xm0
    movq   [dstq+strideq*2], xm1
    movq   [r10 +strideq*0], xm2
    movq   [r10 +strideq*2], xm3
    movhps [dstq+strideq*1], xm0
    movhps [dstq+r9       ], xm1
    movhps [r10 +strideq*1], xm2
    movhps [r10 +r9       ], xm3
    RET
.sec_only:
    movifnidn     t1d, secm
    call .sec
.end_no_clip:
    mova          xm8, [base+end_perm]
    kxnorb         k1, k1, k1
    vpshldd        m4, m0, 8  ; (px << 8) + ((sum > -8) << 4)
    vpshldd        m5, m1, 8
    vpshldd        m6, m2, 8
    vpshldd        m7, m3, 8
    paddw          m0, m4     ; (px << 8) + ((sum + (sum > -8) + 7) << 4)
    paddw          m1, m5
    paddw          m2, m6
    paddw          m3, m7
    vpermb         m0, m8, m0
    vpermb         m1, m8, m1
    vpermb         m2, m8, m2
    vpermb         m3, m8, m3
    add           r10, 2
    punpckldq      m4, m0, m1
    punpckhdq      m0, m1
    punpckldq      m5, m2, m3
    punpckhdq      m2, m3
    movq   [dstq+strideq*0], xm4
    movq   [dstq+strideq*2], xm0
    movq   [r10 +strideq*0], xm5
    movq   [r10 +strideq*2], xm2
    movhps [dstq+strideq*1], xm4
    movhps [dstq+r9       ], xm0
    movhps [r10 +strideq*1], xm5
    movhps [r10 +r9       ], xm2
    RET
.mask_edges_sec_only:
    movifnidn     t1d, secm
    call .mask_edges_sec
    jmp .end_no_clip
ALIGN function_align
.mask_edges:
    mov           t0d, r6d
    mov           t1d, r6d
    or            t0d, 0xA ; top-left 4x4 has bottom and right
    or            t1d, 0x9 ; top-right 4x4 has bottom and left
    vpbroadcastq  m26, [base+edge_mask+t0*8]
    vpbroadcastq  m27, [base+edge_mask+t1*8]
    mov           t1d, r6d
    or            r6d, 0x6 ; bottom-left 4x4 has top and right
    or            t1d, 0x5 ; bottom-right 4x4 has top and left
    vpbroadcastq  m28, [base+edge_mask+r6*8]
    vpbroadcastq  m29, [base+edge_mask+t1*8]
    mov           t0d, dirm
    test         prid, prid
    jz .mask_edges_sec_only
    vpaddd        m20, m30, [base+cdef_dirs+(t0+2)*4] {1to16}
    vpshufbitqmb   k1, m26, m20 ; index in-range
    vpshufbitqmb   k2, m27, m20
    vpshufbitqmb   k3, m28, m20
    vpshufbitqmb   k4, m29, m20
    mova           m8, m4
    mova           m9, m5
    mova          m10, m6
    mova          m11, m7
    vpermb     m8{k1}, m20, m12
    vpermb     m9{k2}, m20, m13
    vpermb    m10{k3}, m20, m14
    vpermb    m11{k4}, m20, m15
    mova   [rsp+0x00], m26
    mova   [rsp+0x40], m27
    mova   [rsp+0x80], m28
    mova   [rsp+0xC0], m29
    CDEF_FILTER_8x8_PRI
    test          t1d, t1d
    jz .end_no_clip
    mova          m26, [rsp+0x00]
    mova          m27, [rsp+0x40]
    mova          m28, [rsp+0x80]
    mova          m29, [rsp+0xC0]
    call .mask_edges_sec
    jmp .end_clip
.mask_edges_sec:
    vpaddd        m20, m30, [base+cdef_dirs+(t0+4)*4] {1to16}
    vpaddd        m21, m30, [base+cdef_dirs+(t0+0)*4] {1to16}
    vpshufbitqmb   k1, m26, m20
    vpshufbitqmb   k2, m27, m20
    vpshufbitqmb   k3, m28, m20
    vpshufbitqmb   k4, m29, m20
    mova          m16, m4
    mova          m17, m5
    mova          m18, m6
    mova          m19, m7
    vpermb    m16{k1}, m20, m12
    vpermb    m17{k2}, m20, m13
    vpermb    m18{k3}, m20, m14
    vpermb    m19{k4}, m20, m15
    vpshufbitqmb   k1, m26, m21
    vpshufbitqmb   k2, m27, m21
    vpshufbitqmb   k3, m28, m21
    vpshufbitqmb   k4, m29, m21
    vpermb        m12, m21, m12
    vpermb        m13, m21, m13
    vpermb        m14, m21, m14
    vpermb        m15, m21, m15
    vpblendmb m12{k1}, m4, m12
    vpblendmb m13{k2}, m5, m13
    vpblendmb m14{k3}, m6, m14
    vpblendmb m15{k4}, m7, m15
    jmp .sec_main
ALIGN function_align
.sec:
    vpaddd        m20, m30, [base+cdef_dirs+(t0+4)*4] {1to16} ; dir + 2
    vpaddd        m21, m30, [base+cdef_dirs+(t0+0)*4] {1to16} ; dir - 2
    vpermb        m16, m20, m12 ; pNtl k0s0 k0s1 k1s0 k1s1
    vpermb        m17, m20, m13 ; pNtr
    vpermb        m18, m20, m14 ; pNbl
    vpermb        m19, m20, m15 ; pNbr
    vpermb        m12, m21, m12 ; pNtl k0s2 k0s3 k1s2 k1s3
    vpermb        m13, m21, m13 ; pNtr
    vpermb        m14, m21, m14 ; pNbl
    vpermb        m15, m21, m15 ; pNbr
.sec_main:
%macro CDEF_FILTER_8x8_SEC 4-5 0 ; load constants
    vpcmpub        k1, m4, %1, 6
    vpcmpub        k2, m5, %2, 6
    vpcmpub        k3, m6, %3, 6
    vpcmpub        k4, m7, %4, 6
    psubb         m20, %1, m4
    psubb         m21, %2, m5
    psubb         m22, %3, m6
    psubb         m23, %4, m7
%if %5
    vpbroadcastb  m28, t1d
    lzcnt         t1d, t1d
    vpbroadcastq  m29, [r3+t1*8]
%endif
    vpsubb    m20{k1}, m4, %1
    vpsubb    m21{k2}, m5, %2
    vpsubb    m22{k3}, m6, %3
    vpsubb    m23{k4}, m7, %4
    gf2p8affineqb m24, m20, m29, 0
    gf2p8affineqb m25, m21, m29, 0
    gf2p8affineqb m26, m22, m29, 0
    gf2p8affineqb m27, m23, m29, 0
%if %5
    vpbroadcastd  m30, [base+sec_tap]
%endif
    psubusb       m24, m28, m24
    psubusb       m25, m28, m25
    psubusb       m26, m28, m26
    psubusb       m27, m28, m27
    pminub        m20, m24
    pminub        m21, m25
    pminub        m22, m26
    pminub        m23, m27
    mova          m24, m30
    mova          m25, m30
    mova          m26, m30
    mova          m27, m30
    vpsubb    m24{k1}, m31, m30
    vpsubb    m25{k2}, m31, m30
    vpsubb    m26{k3}, m31, m30
    vpsubb    m27{k4}, m31, m30
    vpdpbusd       m0, m20, m24
    vpdpbusd       m1, m21, m25
    vpdpbusd       m2, m22, m26
    vpdpbusd       m3, m23, m27
%endmacro
    CDEF_FILTER_8x8_SEC m16, m17, m18, m19, 1
    CDEF_FILTER_8x8_SEC m12, m13, m14, m15
    ret

%endif ; HAVE_AVX512ICL
%endif ; ARCH_X86_64
