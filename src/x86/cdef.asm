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

%include "ext/x86/x86inc.asm"

%if ARCH_X86_64

SECTION_RODATA 32
pd_47130256: dd 4, 7, 1, 3, 0, 2, 5, 6
div_table: dd 840, 420, 280, 210, 168, 140, 120, 105
           dd 420, 210, 140, 105
shufw_6543210x: db 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15
shufb_lohi: db 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
tap_table: ; masks for 8 bit shifts
           db 0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01
           ; weights
           db 4, 2, 3, 3, 2, 1
           db 0, 0 ; padding
           db -1, -2,  0,  0,  1,  2,  0,  0
           db  0, -1,  0,  1,  1,  2,  0,  0
           db  0,  0,  1,  2, -1, -2,  0,  0
           db  0,  1,  1,  2,  0, -1,  0,  0
           db  1,  2,  1,  2,  0,  0,  0,  0
           db  1,  2,  1,  2,  0,  1,  0,  0
           db  1,  2, -1, -2,  1,  2,  0,  0
           db  1,  2,  0, -1,  1,  2,  0,  0

           db  1,  2,  1,  2,  0,  0,  0,  0
           db  1,  2,  1,  2,  0, -1,  0,  0
           db  1,  2,  1,  2,  1,  2,  0,  0
           db  1,  2,  0,  1,  1,  2,  0,  0
           db  1,  2,  0,  0,  1,  2,  0,  0
           db  0,  1,  0, -1,  1,  2,  0,  0
           db  0,  0,  1,  2,  1,  2,  0,  0
           db  0, -1,  1,  2,  0,  1,  0,  0
pw_128: times 2 dw 128
pw_2048: times 2 dw 2048

SECTION .text

; stride unused
%macro ACCUMULATE_TAP 7 ; tap_offset, shift, mask, strength, mul_tap, w, stride
    ; load p0/p1
    movsx         offq, dword [offsets+kq*4+%1] ; off1
%if %6 == 4
    movq           xm5, [tmp0q+offq*2]          ; p0
    movq           xm6, [tmp2q+offq*2]
    movhps         xm5, [tmp1q+offq*2]
    movhps         xm6, [tmp3q+offq*2]
    vinserti128     m5, xm6, 1
%else
    movu           xm5, [tmp0q+offq*2]          ; p0
    vinserti128     m5, [tmp1q+offq*2], 1
%endif
    neg           offq                          ; -off1
%if %6 == 4
    movq           xm6, [tmp0q+offq*2]          ; p1
    movq           xm9, [tmp2q+offq*2]
    movhps         xm6, [tmp1q+offq*2]
    movhps         xm9, [tmp3q+offq*2]
    vinserti128     m6, xm9, 1
%else
    movu           xm6, [tmp0q+offq*2]          ; p1
    vinserti128     m6, [tmp1q+offq*2], 1
%endif
    ; out of bounds values are set to a value that is a both a large unsigned
    ; value and a negative signed value.
    ; use signed max and unsigned min to remove them
    pmaxsw          m7, m5                      ; max after p0
    pminuw          m8, m5                      ; min after p0
    pmaxsw          m7, m6                      ; max after p1
    pminuw          m8, m6                      ; min after p1

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

%macro CDEF_FILTER 2 ; w, h
INIT_YMM avx2
%if %1*%2*2/mmsize > 1
 %if %1 == 4
cglobal cdef_filter_%1x%2, 4, 13, 16, 64
 %else
cglobal cdef_filter_%1x%2, 4, 11, 16, 64
 %endif
%else
cglobal cdef_filter_%1x%2, 4, 12, 16, 64
%endif
%define offsets rsp+32
    DEFINE_ARGS dst, dst_stride, tmp, tmp_stride, pri, sec, pridmp, table, \
                secdmp, damping, dir
    lea         tableq, [tap_table]

    ; off1/2/3[k] [6 total]
    movd           xm2, tmp_strided
    vpbroadcastd    m2, xm2
    mov           dird, r6m
    pmovsxbd        m3, [tableq+dirq*8+16]
    pmulld          m2, m3
    pmovsxbd        m4, [tableq+dirq*8+16+64]
    paddd           m2, m4
    mova     [offsets], m2

    ; register to shuffle values into after packing
    vbroadcasti128 m12, [shufb_lohi]

    movifnidn     prid, prim
    mov       dampingd, r7m
    lzcnt      pridmpd, prid
%if UNIX64
    movd           xm0, prid
    movd           xm1, secd
%endif
    lzcnt      secdmpd, secm
    sub       dampingd, 31
    DEFINE_ARGS dst, dst_stride, tmp, tmp_stride, pri, sec, pridmp, table, \
                secdmp, damping, zero
    xor          zerod, zerod
    add        pridmpd, dampingd
    cmovl      pridmpd, zerod
    add        secdmpd, dampingd
    cmovl      secdmpd, zerod
    mov        [rsp+0], pridmpq                 ; pri_shift
    mov        [rsp+8], secdmpq                 ; sec_shift

    DEFINE_ARGS dst, dst_stride, tmp, tmp_stride, pri, sec, pridmp, table, \
                secdmp
    vpbroadcastb   m13, [tableq+pridmpq]        ; pri_shift_mask
    vpbroadcastb   m14, [tableq+secdmpq]        ; sec_shift_mask

    ; pri/sec_taps[k] [4 total]
    DEFINE_ARGS dst, dst_stride, tmp, tmp_stride, pri, sec, dummy, table, \
                secdmp
%if UNIX64
    vpbroadcastb    m0, xm0                     ; pri_strength
    vpbroadcastb    m1, xm1                     ; sec_strength
%else
    vpbroadcastb    m0, prim
    vpbroadcastb    m1, secm
%endif
    and           prid, 1
    lea           priq, [tableq+priq*2+8]       ; pri_taps
    lea           secq, [tableq+12]             ; sec_taps
%if %1*%2*2/mmsize > 1
 %if %1 == 4
    DEFINE_ARGS dst, dst_stride, tmp0, tmp_stride, pri, sec, dst_stride3, h, off, k, tmp1, tmp2, tmp3
    lea   dst_stride3q, [dst_strideq*3]
 %else
    DEFINE_ARGS dst, dst_stride, tmp0, tmp_stride, pri, sec, h, off, k, tmp1
 %endif
    mov             hd, %1*%2*2/mmsize
%else
    DEFINE_ARGS dst, dst_stride, tmp0, tmp_stride, pri, sec, dst_stride3, off, k, tmp1, tmp2, tmp3
    lea   dst_stride3q, [dst_strideq*3]
%endif
    pxor           m11, m11
%if %1*%2*2/mmsize > 1
.v_loop:
%endif
    lea          tmp1q, [tmp0q+tmp_strideq*2]
%if %1 == 4
    lea          tmp2q, [tmp0q+tmp_strideq*4]
    lea          tmp3q, [tmp1q+tmp_strideq*4]
%endif
    mov             kd, 1
%if %1 == 4
    movq           xm4, [tmp0q]
    movhps         xm4, [tmp1q]
    movq           xm5, [tmp2q]
    movhps         xm5, [tmp3q]
    vinserti128     m4, xm5, 1
%else
    mova           xm4, [tmp0q]                  ; px
    vinserti128     m4, [tmp1q], 1
%endif
    pxor           m15, m15                     ; sum
    mova            m7, m4                      ; max
    mova            m8, m4                      ; min
.k_loop:
    vpbroadcastb    m2, [priq+kq]               ; pri_taps
    vpbroadcastb    m3, [secq+kq]               ; sec_taps

    ACCUMULATE_TAP 0*8, [rsp+0], m13, m0, m2, %1, %3
    ACCUMULATE_TAP 1*8, [rsp+8], m14, m1, m3, %1, %3
    ACCUMULATE_TAP 2*8, [rsp+8], m14, m1, m3, %1, %3

    dec             kq
    jge .k_loop

    vpbroadcastd   m10, [pw_2048]
    pcmpgtw         m9, m11, m15
    paddw          m15, m9
    pmulhrsw       m15, m10
    paddw           m4, m15
    pminsw          m4, m7
    pmaxsw          m4, m8
    packuswb        m4, m4
    vextracti128   xm5, m4, 1
%if %1 == 4
    movd [dstq+dst_strideq*0], xm4
    pextrd [dstq+dst_strideq*1], xm4, 1
    movd [dstq+dst_strideq*2], xm5
    pextrd [dstq+dst_stride3q], xm5, 1
%else
    movq [dstq+dst_strideq*0], xm4
    movq [dstq+dst_strideq*1], xm5
%endif

%if %1*%2*2/mmsize > 1
 %define vloop_lines (mmsize/(%1*2))
    lea           dstq, [dstq+dst_strideq*vloop_lines]
    lea          tmp0q, [tmp0q+tmp_strideq*2*vloop_lines]
    dec             hd
    jg .v_loop
%endif

    RET
%endmacro

CDEF_FILTER 8, 8
CDEF_FILTER 4, 8
CDEF_FILTER 4, 4

INIT_YMM avx2
cglobal cdef_dir, 3, 4, 15, src, stride, var, stride3
    ; Instead of loading 8-bit values and unpacking like dav1d, values are
    ; already converted to 16-bit.
    lea       stride3q, [strideq*3]
    mova           xm0, [srcq+strideq*0]
    mova           xm1, [srcq+strideq*1]
    mova           xm2, [srcq+strideq*2]
    mova           xm3, [srcq+stride3q]
    lea           srcq, [srcq+strideq*4]
    vinserti128     m3, [srcq+strideq*0], 1
    vinserti128     m2, [srcq+strideq*1], 1
    vinserti128     m1, [srcq+strideq*2], 1
    vinserti128     m0, [srcq+stride3q], 1
    vpbroadcastd    m8, [pw_128]

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
%endif ; ARCH_X86_64
