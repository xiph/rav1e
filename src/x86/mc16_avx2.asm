; Copyright (c) 2017-2020, The rav1e contributors
; Copyright (c) 2020, Nathan Egge
; All rights reserved.
;
; This source code is subject to the terms of the BSD 2 Clause License and
; the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
; was not distributed with this source code in the LICENSE file, you can
; obtain it at www.aomedia.org/license/software. If the Alliance for Open
; Media Patent License 1.0 was not distributed with this source code in the
; PATENTS file, you can obtain it at www.aomedia.org/license/patent.

%include "config.asm"
%include "ext/x86/x86inc.asm"

%if ARCH_X86_64

SECTION_RODATA 32

spf_h_shuf: db  0,  1,  2,  3,  4,  5,  6,  7,  2,  3,  4,  5,  6,  7,  8,  9
            db  4,  5,  6,  7,  8,  9, 10, 11,  6,  7,  8,  9, 10, 11, 12, 13
pq_2:       dq (6 - 4)
pq_4:       dq (6 - 2)
pq_8:       dq (6 + 2)
pq_10:      dq (6 + 4)
pd_32:      dd (1 << 6 >> 1)
pd_34:      dd (1 << 6 >> 1) + (1 << (6 - 4) >> 1)
pd_40:      dd (1 << 6 >> 1) + (1 << (6 - 2) >> 1)
pd_2:       dd (1 << (6 - 4) >> 1)
pd_512:     dd (1 << (6 + 4) >> 1)
pd_8:       dd (1 << (6 - 2) >> 1)
pd_128:     dd (1 << (6 + 2) >> 1)
nd_524256:  dd (1 << 6 >> 1) - (8192 << 6)
nd_32766:   dd (1 << (6 - 4) >> 1) - (8192 << (6 - 4))
nd_131064:  dd (1 << (6 - 2) >> 1) - (8192 << (6 - 2))
pw_8192:    dw 8192

SECTION .text

%macro PUT_4TAP_H 6
  pshufb %1, %3
  pshufb %2, %3
  pmaddwd %1, %4
  pmaddwd %2, %4
  phaddd %1, %2
  paddd %1, %5
  psrad %1, %6
%endm

%macro PUT_8TAP_H 8
  movu xm%1, [srcq + %8 + 0]
  movu xm%3, [srcq + %8 + 2]
  vinserti128 m%1, [srcq + ssq + %8 + 0], 1
  vinserti128 m%3, [srcq + ssq + %8 + 2], 1
  movu xm%2, [srcq + %8 + 4]
  movu xm%4, [srcq + %8 + 6]
  vinserti128 m%2, [srcq + ssq + %8 + 4], 1
  vinserti128 m%4, [srcq + ssq + %8 + 6], 1
  pmaddwd m%1, %5
  pmaddwd m%3, %5
  pmaddwd m%2, %5
  pmaddwd m%4, %5
  phaddd m%1, m%3
  phaddd m%2, m%4
  phaddd m%1, m%2
  paddd m%1, %6
  psrad m%1, %7
%endm

%macro PUT_4TAP_HS1 5
  pshufb %1, %2
  pmaddwd %1, %3
  phaddd %1, %1
  paddd %1, %4
  psrad %1, %5
  packssdw %1, %1
%endm

%macro PUT_4TAP_HS2 6
  pshufb %1, %3
  pshufb %2, %3
  pmaddwd %1, %4
  pmaddwd %2, %4
  phaddd %1, %1
  phaddd %2, %2
  paddd %1, %5
  paddd %2, %5
  psrad %1, %6
  psrad %2, %6
  packssdw %1, %1
  packssdw %2, %2
%endm

%macro PUT_8TAP_HS 7-8
  movu xm%1, [srcq + %7 + 0]
  movu xm%3, [srcq + %7 + 2]
  vinserti128 m%1, [srcq + %7 + 8], 1
  vinserti128 m%3, [srcq + %7 + 10], 1
  pmaddwd m%1, %4
  pmaddwd m%3, %4
  phaddd m%1, m%3
  movu xm%2, [srcq + %7 + 4]
  movu xm%3, [srcq + %7 + 6]
  vinserti128 m%2, [srcq + %7 + 12], 1
  vinserti128 m%3, [srcq + %7 + 14], 1
  pmaddwd m%2, %4
  pmaddwd m%3, %4
  phaddd m%2, m%3
%if %0 > 7
  vpbroadcastd %5, %8
%endif
  phaddd m%1, m%2
  paddd m%1, %5
  psrad m%1, %6
  packssdw m%1, m%1
%endm

%macro LOAD_REGS_2 3
  mov%1 xm%2, [srcq + ssq*0]
  mov%1 xm%3, [srcq + ssq*1]
%ifidn %1, u
  vpermq m%2, m%2, q3120
  vpermq m%3, m%3, q3120
%endif
  lea srcq, [srcq + ssq*2]
%endm

%macro LOAD_REGS_3 4
  mov%1 xm%2, [srcq + ssq*0]
  mov%1 xm%3, [srcq + ssq*1]
  mov%1 xm%4, [srcq + ssq*2]
%ifidn %1, u
  vpermq m%2, m%2, q3120
  vpermq m%3, m%3, q3120
  vpermq m%4, m%4, q3120
%endif
  add srcq, ss3q
%endm

%macro LOAD_REGS 3-8
%if %0 == 3
  LOAD_REGS_2 %1, %2, %3
%elif %0 == 4
  LOAD_REGS_3 %1, %2, %3, %4
%elif %0 == 5
  LOAD_REGS_2 %1, %2, %3
  LOAD_REGS_2 %1, %4, %5
%elif %0 == 6
  LOAD_REGS_3 %1, %2, %3, %4
  LOAD_RESG_2 %1, %5, %6
%elif %0 == 7
  LOAD_REGS_3 %1, %2, %3, %4
  LOAD_REGS_3 %1, %5, %6, %7
%else
  LOAD_REGS_3 %1, %2, %3, %4
  LOAD_REGS_2 %1, %5, %6
  LOAD_REGS_2 %1, %7, %8
%endif
%endm

%macro STORE_REGS 3
%ifidn %1, u
  vpermq m%2, m%2, q3120
  vpermq m%3, m%3, q3120
%endif
  mov%1 [dstq + dsq*0], xm%2
  mov%1 [dstq + dsq*1], xm%3
  lea dstq, [dstq + dsq*2]
%endm

%macro INTERLEAVE_REGS 4-8
  punpckl%1 %2, %3
  punpckl%1 %3, %4
%if %0 > 4
  punpckl%1 %4, %5
  punpckl%1 %5, %6
%endif
%if %0 > 6
  punpckl%1 %6, %7
  punpckl%1 %7, %8
%endif
%endm

%macro MUL_ADD_R 8
  pmaddwd %3, %7
  pmaddwd %1, %5, %8
  paddd %1, %3
  mova %3, %5
  pmaddwd %4, %7
  pmaddwd %2, %6, %8
  paddd %2, %4
  mova %4, %6
%endm

%macro MUL_ACC_R 7
  pmaddwd %3, %5, %7
  pmaddwd %4, %6, %7
  paddd %1, %3
  paddd %2, %4
  mova %3, %5
  mova %4, %6
%endm

%macro RND_SHR_MIN_R 5
  paddd %1, %3
  paddd %2, %3
  psrad %1, %4
  psrad %2, %4
  packusdw %1, %1
  packusdw %2, %2
  pminuw %1, %5
  pminuw %2, %5
%endm

%macro RND_SHR_R 4
  paddd %1, %3
  paddd %2, %3
  psrad %1, %4
  psrad %2, %4
  packssdw %1, %1
  packssdw %2, %2
%endm

; int8_t subpel_filters[5][15][8]
%assign FILTER_REGULAR (0*15 << 7) | 3*15
%assign FILTER_SMOOTH  (1*15 << 7) | 4*15
%assign FILTER_SHARP   (2*15 << 7) | 3*15

%macro make_8tap_fn 4 ; type, op, type_h, type_v
INIT_XMM avx2
cglobal %1_8tap_%2_16bpc
  mov t0d, FILTER_%3
  mov t1d, FILTER_%4
  jmp mangle(private_prefix %+ _%1_8tap_16bpc %+ SUFFIX)
%endmacro

cextern mc_subpel_filters

%define subpel_filters (mangle(private_prefix %+ _mc_subpel_filters)-8)

%macro filter_fn 1

%if WIN64
%ifidn %1, put
DECLARE_REG_TMP 5, 4
%else
DECLARE_REG_TMP 4, 5
%endif
%else
DECLARE_REG_TMP 7, 8
%endif

make_8tap_fn %1, regular,        REGULAR, REGULAR
make_8tap_fn %1, regular_smooth, REGULAR, SMOOTH
make_8tap_fn %1, regular_sharp,  REGULAR, SHARP
make_8tap_fn %1, smooth,         SMOOTH,  SMOOTH
make_8tap_fn %1, smooth_regular, SMOOTH,  REGULAR
make_8tap_fn %1, smooth_sharp,   SMOOTH,  SHARP
make_8tap_fn %1, sharp,          SHARP,   SHARP
make_8tap_fn %1, sharp_regular,  SHARP,   REGULAR
make_8tap_fn %1, sharp_smooth,   SHARP,   SMOOTH

INIT_YMM avx2
%ifidn %1, put
cglobal put_8tap_16bpc, 4, 10, 16, dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%else
cglobal prep_8tap_16bpc, 3, 10, 16, dst, src, ss, _w, h, mx, my, bdmax, ds, ss3
%endif

%ifidn %1, put
  imul mxd, mxm, 0x4081 ; (1 << 14) | (1 << 7) | (1 << 0)
  add mxd, t0d
  imul myd, mym, 0x4081 ; (1 << 14) | (1 << 7) | (1 << 0)
  add myd, t1d
%else
  imul myd, mym, 0x4081 ; (1 << 14) | (1 << 7) | (1 << 0)
  add myd, t1d
  imul mxd, mxm, 0x4081 ; (1 << 14) | (1 << 7) | (1 << 0)
  add mxd, t0d
%endif

  movsxd _wq, _wm
  movifnidn hd, hm

%ifidn %1, put
  vpbroadcastw m7, bdmaxm
%else
  lea dsq, [_wq*2]
%endif

  test mxd, (0x7f << 14)
  jnz .%1_8tap_h_16bpc
  test myd, (0x7f << 14)
  jnz .%1_8tap_v_16bpc

; ---- {put,prep}_16bpc ----

INIT_XMM avx2
.%1_16bpc: ; cglobal put_16bpc, 6, 8, 8, dst, ds, src, ss, w, h

%ifidn %1, prep
INIT_YMM avx2
  popcnt bdmaxd, bdmaxm
  vpbroadcastq m8, [pq_4]
  vpbroadcastw m9, [pw_8192]
  cmp bdmaxd, 12
  jne .prep_bits10
  vpbroadcastq m8, [pq_2]
.prep_bits10:
INIT_XMM avx2
%endif

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, jr, ss3
%else
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, jr, ds, ss3
%endif

  lea jrq, [.jmp_tbl]
  tzcnt _wd, _wm
%ifidn %1, put
  sub _wd, 1
%else
  sub _wd, 2
%endif
  movsxd _wq, [jrq + _wq*4]
  add _wq, jrq
  jmp _wq

%ifidn %1, put
.w2: ; 2xN
  movd m0, [srcq]
  movd m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]
  movd [dstq], m0
  movd [dstq + dsq], m1
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w2
  RET
%endif

.w4: ; 4xN
  movq m0, [srcq]
  movq m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]
%ifidn %1, prep
  psllw m0, m8
  psllw m1, m8
  psubw m0, m9
  psubw m1, m9
%endif
  movq [dstq], m0
  movq [dstq + dsq], m1
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w4
  RET

  ; XXX is unaligned input (but aligned output) a hard requirement, or is checkasm broken?
.w8: ; 8xN
  movu m0, [srcq]
  movu m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]
%ifidn %1, prep
  psllw m0, m8
  psllw m1, m8
  psubw m0, m9
  psubw m1, m9
%endif
  mova [dstq], m0
  mova [dstq + dsq], m1
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w8
  RET

INIT_YMM avx2
.w16: ; 16xN
  movu m0, [srcq]
  movu m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]
%ifidn %1, prep
  psllw m0, xm8
  psllw m1, xm8
  psubw m0, m9
  psubw m1, m9
%endif
  mova [dstq], m0
  mova [dstq + dsq], m1
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w16
  RET

.w32: ; 32xN
  movu m0, [srcq + 32*0]
  movu m1, [srcq + 32*1]
  movu m2, [srcq + ssq]
  movu m3, [srcq + ssq + 32*1]
  lea srcq, [srcq + ssq*2]
%ifidn %1, prep
  psllw m0, xm8
  psllw m1, xm8
  psllw m2, xm8
  psllw m3, xm8
  psubw m0, m9
  psubw m1, m9
  psubw m2, m9
  psubw m3, m9
%endif
  mova [dstq + 32*0], m0
  mova [dstq + 32*1], m1
  mova [dstq + dsq + 32*0], m2
  mova [dstq + dsq + 32*1], m3
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w32
  RET

.w64: ; 64xN
  movu m0, [srcq + 32*0]
  movu m1, [srcq + 32*1]
  movu m2, [srcq + 32*2]
  movu m3, [srcq + 32*3]
  movu m4, [srcq + ssq + 32*0]
  movu m5, [srcq + ssq + 32*1]
  movu m6, [srcq + ssq + 32*2]
  movu m7, [srcq + ssq + 32*3]
  lea srcq, [srcq + ssq*2]
%ifidn %1, prep
  psllw m0, xm8
  psllw m1, xm8
  psllw m2, xm8
  psllw m3, xm8
  psllw m4, xm8
  psllw m5, xm8
  psllw m6, xm8
  psllw m7, xm8
  psubw m0, m9
  psubw m1, m9
  psubw m2, m9
  psubw m3, m9
  psubw m4, m9
  psubw m5, m9
  psubw m6, m9
  psubw m7, m9
%endif
  mova [dstq + 32*0], m0
  mova [dstq + 32*1], m1
  mova [dstq + 32*2], m2
  mova [dstq + 32*3], m3
  mova [dstq + dsq + 32*0], m4
  mova [dstq + dsq + 32*1], m5
  mova [dstq + dsq + 32*2], m6
  mova [dstq + dsq + 32*3], m7
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .w64
  RET

.w128: ; 128xN
  movu m0, [srcq + 32*0]
  movu m1, [srcq + 32*1]
  movu m2, [srcq + 32*2]
  movu m3, [srcq + 32*3]
  movu m4, [srcq + 32*4]
  movu m5, [srcq + 32*5]
  movu m6, [srcq + 32*6]
  movu m7, [srcq + 32*7]
  add srcq, ssq
%ifidn %1, prep
  psllw m0, xm8
  psllw m1, xm8
  psllw m2, xm8
  psllw m3, xm8
  psllw m4, xm8
  psllw m5, xm8
  psllw m6, xm8
  psllw m7, xm8
  psubw m0, m9
  psubw m1, m9
  psubw m2, m9
  psubw m3, m9
  psubw m4, m9
  psubw m5, m9
  psubw m6, m9
  psubw m7, m9
%endif
  mova [dstq + 32*0], m0
  mova [dstq + 32*1], m1
  mova [dstq + 32*2], m2
  mova [dstq + 32*3], m3
  mova [dstq + 32*4], m4
  mova [dstq + 32*5], m5
  mova [dstq + 32*6], m6
  mova [dstq + 32*7], m7
  add dstq, dsq
  dec hd
  jg .w128
  RET

.jmp_tbl:
%ifidn %1, put
  dd .w2 - .jmp_tbl
%endif
  dd .w4 - .jmp_tbl
  dd .w8 - .jmp_tbl
  dd .w16 - .jmp_tbl
  dd .w32 - .jmp_tbl
  dd .w64 - .jmp_tbl
  dd .w128 - .jmp_tbl

; ---- {put,prep}_8tap_h_16bpc ----

INIT_XMM avx2
.%1_8tap_h_16bpc: ; cglobal put_8tap_h_16bpc, 4, 9, 0, dst, ds, src, ss, w, h, mx, my, bdmax
%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%else
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, bdmax, ds, ss3
%endif

  cmp _wd, 4
  jle .h_use4tap
  shr mxd, 7
.h_use4tap:
  and mxd, 0x7f

  test myd, (0x7f << 14)
  jnz .%1_8tap_hv_16bpc

INIT_YMM avx2
  popcnt bdmaxd, bdmaxm
%ifidn %1, put
  vpbroadcastd m6, [pd_34]    ; (1 << 6 >> 1) + (1 << (6 - 4) >> 1)
%else
  vpbroadcastd m6, [nd_32766] ; (1 << (6 - 4) >> 1) - (8192 << (6 - 4))
  vpbroadcastq m7, [pq_2]     ; (6 - 4)
%endif
  cmp bdmaxd, 12
  jne .h_bits10
%ifidn %1, put
  vpbroadcastd m6, [pd_40]     ; (1 << 6 >> 1) + (1 << (6 - 2) >> 1)
%else
  vpbroadcastd m6, [nd_131064] ; (1 << (6 - 2) >> 1) - (8192 << (6 - 2))
  vpbroadcastq m7, [pq_4]      ; (6 - 2)
%endif
.h_bits10:
INIT_XMM avx2

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, w2, jr, ss3
%else
  DEFINE_ARGS dst, src, ss, _w, h, mx, w2, jr, ds, ss3
%endif

  lea w2q, [_wq*2]

  lea jrq, [.h_jmp_tbl]
  tzcnt _wd, _wm
%ifidn %1, put
  sub _wd, 1
%else
  sub _wd, 2
%endif
  movsxd _wq, [jrq + _wq*4]
  add _wq, jrq
  jmp _wq

%ifidn %1, put
.h_w2:
  sub srcq, 2
  mova xm4, [spf_h_shuf]
  vpbroadcastd m5, [jrq - .h_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, m5

.h_w2l:
  movu m0, [srcq]
  movu m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

%ifidn %1, put
  PUT_4TAP_H m0, m1, m4, m5, m6, 6
  packusdw m0, m0
  pminuw m0, m7
%else
  PUT_4TAP_H m0, m1, m4, m5, m6, m7
  packssdw m0, m1
%endif

  movd [dstq], m0
  pextrd [dstq + dsq], m0, 1
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .h_w2l
  RET
%endif

INIT_YMM avx2
.h_w4:
  sub srcq, 2
  mova m4, [spf_h_shuf]
  vpbroadcastd xm5, [jrq - .h_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, xm5

.h_w4l:
  vbroadcasti128 m0, [srcq]
  vbroadcasti128 m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

%ifidn %1, put
  PUT_4TAP_H m0, m1, m4, m5, m6, 6
  packusdw m0, m0
  pminuw m0, m7
%else
  PUT_4TAP_H m0, m1, m4, m5, m6, xm7
  packssdw m0, m0
%endif

  vextracti128 xm1, m0, 1
  movd [dstq], xm0
  movd [dstq + 4], xm1
  pextrd [dstq + dsq], xm0, 1
  pextrd [dstq + dsq + 4], xm1, 1
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .h_w4l
  RET

.h_w8:
  sub srcq, 6
  vpbroadcastq xm5, [jrq - .h_jmp_tbl + subpel_filters + mxq*8]
  vpmovsxbw m5, xm5

.h_w8l:
  mov _wd, w2d

.h_w8c:
%ifidn %1, put
  PUT_8TAP_H 0, 1, 2, 3, m5, m6, 6, 4*0
  PUT_8TAP_H 1, 2, 3, 4, m5, m6, 6, 4*2
  packusdw m0, m1
  pminuw m0, m7
%else
  PUT_8TAP_H 0, 1, 2, 3, m5, m6, xm7, 4*0
  PUT_8TAP_H 1, 2, 3, 4, m5, m6, xm7, 4*2
  packssdw m0, m1
%endif
  add srcq, 8*2

  mova [dstq], xm0
  vextracti128 [dstq + dsq], m0, 1

  add dstq, 8*2
  sub _wd, 8*2
  jg .h_w8c

  sub srcq, w2q
  sub dstq, w2q
  lea srcq, [srcq + ssq*2]
  lea dstq, [dstq + dsq*2]
  sub hd, 2
  jg .h_w8l
  RET

.h_jmp_tbl:
%ifidn %1, put
  dd .h_w2 - .h_jmp_tbl
%endif
  dd .h_w4 - .h_jmp_tbl
  dd .h_w8 - .h_jmp_tbl
  dd .h_w8 - .h_jmp_tbl
  dd .h_w8 - .h_jmp_tbl
  dd .h_w8 - .h_jmp_tbl
  dd .h_w8 - .h_jmp_tbl

; ---- {put,prep}_8tap_v_16bpc ----

INIT_XMM avx2
.%1_8tap_v_16bpc: ; cglobal put_8tap_v_16bpc, 4, 9, 0, dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%else
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, bdmax, ds, ss3
%endif

  cmp hd, 4
  jle .v_use4tap
  shr myd, 7
.v_use4tap:
  and myd, 0x7f

INIT_YMM avx2
%ifidn %1, put
  vpbroadcastd m6, [pd_32]    ; (1 << 6 >> 1)
%else
  popcnt bdmaxd, bdmaxm
  vpbroadcastd m6, [nd_32766] ; (1 << (6 - 4) >> 1) - (8192 << (6 - 4))
  vpbroadcastq m7, [pq_2]     ; (6 - 4)
  cmp bdmaxd, 12
  jne .v_bits10
  vpbroadcastd m6, [nd_131064] ; (1 << (6 - 2) >> 1) - (8192 << (6 - 2))
  vpbroadcastq m7, [pq_4]      ; (6 - 2)
.v_bits10:
%endif
INIT_XMM avx2

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, w2, my, jr, ss3
%else
  DEFINE_ARGS dst, src, ss, _w, h, w2, my, jr, ds, ss3
%endif

  lea jrq, [.v_jmp_tbl]
  lea w2q, [_wq*2]
  lea ss3q, [ssq*3]

INIT_YMM avx2
  lea myq, [jrq - .v_jmp_tbl + subpel_filters + myq*8]
  vpbroadcastw m8, [myq+0]
  vpbroadcastw m9, [myq+2]
  vpbroadcastw m10, [myq+4]
  vpbroadcastw m11, [myq+6]
  vpmovsxbw m8, xm8
  vpmovsxbw m9, xm9
  vpmovsxbw m10, xm10
  vpmovsxbw m11, xm11
INIT_XMM avx2

  tzcnt _wd, _wm
%ifidn %1, put
  sub _wd, 1
%else
  sub _wd, 2
%endif
  movsxd _wq, [jrq + _wq*4]
  add _wq, jrq
  jmp _wq

%ifidn %1, put
.v_w2:

  cmp hd, 4
  jg .v_w28

  sub srcq, ssq
  LOAD_REGS d, 0, 1, 2
  INTERLEAVE_REGS wd, m0, m1, m2

.v_w2l: ; 2x2, 2x4

  LOAD_REGS d, 3, 4
  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m5, m8, m0, m1, m2, m3, m9, m10
  mova m2, m4

  RND_SHR_MIN_R m5, m8, m6, 6, m7
  STORE_REGS d, 5, 8

  sub hd, 2
  jg .v_w2l
  RET

.v_w28:

  sub srcq, ss3q
  LOAD_REGS d, 0, 1, 2, 3, 4, 12, 13
  INTERLEAVE_REGS wd, m0, m1, m2, m3, m4, m12, m13

.v_w28l: ; 2x6, 2x8, 2x12, 2x16, 2x24, 2x32

  sub srcq, ssq
  LOAD_REGS d, 13, 14, 15
  INTERLEAVE_REGS wd, m13, m14, m15

  MUL_ADD_R m5, m15, m0, m1, m2, m3, m8, m9
  MUL_ACC_R m5, m15, m2, m3, m4, m12, m10
  MUL_ACC_R m5, m15, m4, m12, m13, m14, m11

  RND_SHR_MIN_R m5, m15, m6, 6, m7
  STORE_REGS d, 5, 15

  sub hd, 2
  jg .v_w28l
  RET
%endif

.v_w4:

  cmp hd, 4
  jg .v_w48

  sub srcq, ssq
  LOAD_REGS q, 0, 1, 2
  INTERLEAVE_REGS wd, m0, m1, m2

.v_w4l: ; 4x2 4x4

  LOAD_REGS q, 3, 4
  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m5, m8, m0, m1, m2, m3, m9, m10
  mova m2, m4

%ifidn %1, put
  RND_SHR_MIN_R m5, m8, m6, 6, m7
%else
  RND_SHR_R m5, m8, m6, m7
%endif
  STORE_REGS q, 5, 8

  sub hd, 2
  jg .v_w4l
  RET

.v_w48:

  sub srcq, ss3q
  LOAD_REGS q, 0, 1, 2, 3, 4, 12, 13
  INTERLEAVE_REGS wd, m0, m1, m2, m3, m4, m12, m13

.v_w48l: ; 4x6, 4x8, 4x12, 4x16, 4x24, 4x32

  sub srcq, ssq
  LOAD_REGS q, 13, 14, 15
  INTERLEAVE_REGS wd, m13, m14, m15

  MUL_ADD_R m5, m15, m0, m1, m2, m3, m8, m9
  MUL_ACC_R m5, m15, m2, m3, m4, m12, m10
  MUL_ACC_R m5, m15, m4, m12, m13, m14, m11

%ifidn %1, put
  RND_SHR_MIN_R m5, m15, m6, 6, m7
%else
  RND_SHR_R m5, m15, m6, m7
%endif
  STORE_REGS q, 5, 15

  sub hd, 2
  jg .v_w48l

  RET

INIT_YMM avx2
.v_w8:

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, oh, h, w2, tdst, tsrc, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, oh, h, w2, tdst, tsrc, ds, ss3
%endif

  mov ohd, hd
  mov tdstq, dstq

  cmp hd, 4
  jg .v_w88

  sub srcq, ssq
  mov tsrcq, srcq

.v_w8l: ; N = 8, 16, 32, 64, 128

  LOAD_REGS u, 0, 1, 2
  INTERLEAVE_REGS wd, m0, m1, m2

.v_w8c: ; Nx2, Nx4

  LOAD_REGS u, 3, 4
  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m5, m8, m0, m1, m2, m3, m9, m10
  mova m2, m4

%ifidn %1, put
  RND_SHR_MIN_R m5, m8, m6, 6, m7
%else
  RND_SHR_R m5, m8, m6, xm7
%endif
  STORE_REGS u, 5, 8

  sub hd, 2
  jg .v_w8c

  add tdstq, 2*8
  add tsrcq, 2*8
  mov hd, ohd
  mov dstq, tdstq
  mov srcq, tsrcq
  sub w2d, 2*8
  jg .v_w8l

  RET

.v_w88:

  sub srcq, ss3q
  mov tsrcq, srcq

.v_w88l: ; N = 8, 16, 32, 64, 128

  LOAD_REGS u, 0, 1, 2, 3, 4, 12, 13
  INTERLEAVE_REGS wd, m0, m1, m2, m3, m4, m12, m13

.v_w88c: ; Nx6, Nx8, Nx12, Nx16, Nx24, Nx32

  sub srcq, ssq

  LOAD_REGS u, 13, 14, 15
  INTERLEAVE_REGS wd, m13, m14, m15

  MUL_ADD_R m5, m15, m0, m1, m2, m3, m8, m9
  MUL_ACC_R m5, m15, m2, m3, m4, m12, m10
  MUL_ACC_R m5, m15, m4, m12, m13, m14, m11

%ifidn %1, put
  RND_SHR_MIN_R m5, m15, m6, 6, m7
%else
  RND_SHR_R m5, m15, m6, xm7
%endif
  STORE_REGS u, 5, 15

  sub hd, 2
  jg .v_w88c

  add tdstq, 2*8
  add tsrcq, 2*8
  mov hd, ohd
  mov dstq, tdstq
  mov srcq, tsrcq
  sub w2d, 2*8
  jg .v_w88l

  RET

.v_jmp_tbl:
%ifidn %1, put
  dd .v_w2 - .v_jmp_tbl
%endif
  dd .v_w4 - .v_jmp_tbl
  dd .v_w8 - .v_jmp_tbl
  dd .v_w8 - .v_jmp_tbl
  dd .v_w8 - .v_jmp_tbl
  dd .v_w8 - .v_jmp_tbl
  dd .v_w8 - .v_jmp_tbl

; ---- {put,prep}_8tap_hv_16bpc ----

INIT_XMM avx2
.%1_8tap_hv_16bpc: ; cglobal put_8tap_hv_16bpc, 4, 9, 0, dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, bdmax, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, bdmax, ds, ss3
%endif

  cmp hd, 4
  jle .hv_use4tap
  shr myd, 7
.hv_use4tap:
  and myd, 0x7f

INIT_YMM avx2
  popcnt bdmaxd, bdmaxm
  vpbroadcastd m6, [pd_2]       ; (1 << (6 - 4) >> 1)
  movq xm13, [pq_2]             ; 6 - 4
%ifidn %1, put
  vpbroadcastd m14, [pd_512]    ; (1 << (6 + 4) >> 1)
  movq xm15, [pq_10]            ; 6 + 4
%else
  vpbroadcastd m14, [nd_524256] ; (1 << 6 >> 1) - (8192 << 6)
%endif
  cmp bdmaxd, 12
  jne .hv_bits10
  vpbroadcastd m6, [pd_8]       ; (1 << (6 - 2) >> 1)
  movq xm13, [pq_4]             ; 6 - 2
%ifidn %1, put
  vpbroadcastd m14, [pd_128]    ; (1 << (6 + 2) >> 1)
  movq xm15, [pq_8]             ; 6 + 2
%endif
.hv_bits10:
INIT_XMM avx2

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, jr, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, jr, ds, ss3
%endif

  lea jrq, [.hv_jmp_tbl]

INIT_YMM avx2
  lea ss3q, [jrq - .hv_jmp_tbl + subpel_filters + myq*8]
  vpbroadcastw xm8, [ss3q]
  vpbroadcastw xm9, [ss3q + 2]
  vpbroadcastw xm10, [ss3q + 4]
  vpbroadcastw xm11, [ss3q + 6]
  vpmovsxbw m8, xm8
  vpmovsxbw m9, xm9
  vpmovsxbw m10, xm10
  vpmovsxbw m11, xm11
INIT_XMM avx2

  ; Width is need for for filters 8 and larger, see .hv_w8
  mov ss3q, _wq

  tzcnt _wd, _wm
%ifidn %1, put
  sub _wd, 1
%else
  sub _wd, 2
%endif
  movsxd _wq, [jrq + _wq*4]
  add _wq, jrq
  jmp _wq

%ifidn %1, put
.hv_w2:
  cmp hd, 4
  jg .hv_w28

  lea ss3q, [ssq*3]

  mova m8, [spf_h_shuf]
  vpbroadcastd m5, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, m5

  sub srcq, 2
  sub srcq, ssq

  movu m0, [srcq]
  movu m1, [srcq + ssq]
  movu m2, [srcq + ssq*2]
  add srcq, ss3q

  PUT_4TAP_HS2 m0, m1, m8, m5, m6, m13
  PUT_4TAP_HS1 m2, m8, m5, m6, m13
  INTERLEAVE_REGS wd, m0, m1, m2

.hv_w2l:

  movu m3, [srcq]
  movu m4, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m3, m4, m8, m5, m6, m13

  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m11, m12, m0, m1, m2, m3, m9, m10
  mova m2, m4

  RND_SHR_MIN_R m11, m12, m14, m15, m7
  STORE_REGS d, 11, 12

  sub hd, 2
  jg .hv_w2l

  RET

.hv_w28:
  lea ss3q, [ssq*3]

  mova m8, [spf_h_shuf]
  vpbroadcastd m5, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, m5

  lea myq, [jrq - .hv_jmp_tbl + subpel_filters + myq*8]
  vpbroadcastd m9, [myq]
  vpbroadcastd m10, [myq + 4]
  vpmovsxbw m9, m9
  vpmovsxbw m10, m10

  sub srcq, 2
  sub srcq, ss3q

  movu m0, [srcq]
  movu m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m0, m1, m8, m5, m6, m13

  movu m4, [srcq]
  movu m3, [srcq + ssq]
  movu m2, [srcq + ssq*2]
  add srcq, ss3q

  PUT_4TAP_HS2 m4, m3, m8, m5, m6, m13
  PUT_4TAP_HS1 m2,     m8, m5, m6, m13

  INTERLEAVE_REGS wd, m0, m1, m4, m3, m2
  punpckldq m0, m4
  punpckldq m1, m3

  movu m3, [srcq]
  movu m4, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m3, m4, m8, m5, m6, m13

  INTERLEAVE_REGS wd, m2, m3, m4

.hv_w28l:

  movu m11, [srcq]
  movu m12, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m11, m12, m8, m5, m6, m13

  INTERLEAVE_REGS wd, m4, m11, m12
  punpckldq m2, m4
  punpckldq m3, m11

  pmaddwd m11, m0, m9
  pmaddwd m4, m2, m10
  pmaddwd m12, m1, m9
  paddd m11, m4
  pmaddwd m4, m3, m10
  paddd m12, m4
  phaddd m11, m11
  phaddd m12, m12

  RND_SHR_MIN_R m11, m12, m14, m15, m7
  STORE_REGS d, 11, 12

  pshufd m0, m0, q2031
  pshufd m1, m1, q2031
  pshufd m11, m2, q3120
  pshufd m12, m3, q3120
  pshufd m2, m2, q2031
  pshufd m3, m3, q2031

  mova m4, m3
  psrad m4, 16
  packssdw m4, m4

  punpckldq m0, m11
  punpckldq m1, m12

  sub hd, 2
  jg .hv_w28l

  RET
%endif

INIT_YMM avx2
.hv_w4:
  cmp hd, 4
  jg .hv_w48

  lea ss3q, [ssq*3]

  mova m8, [spf_h_shuf]
  vpbroadcastd xm5, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, xm5

  sub srcq, 2
  sub srcq, ssq

  vbroadcasti128 m0, [srcq]
  vbroadcasti128 m1, [srcq + ssq]
  vbroadcasti128 m2, [srcq + ssq*2]
  add srcq, ss3q

  PUT_4TAP_HS2 m0, m1, m8, m5, m6, xm13
  PUT_4TAP_HS1 m2,     m8, m5, m6, xm13
  INTERLEAVE_REGS wd, m0, m1, m2

.hv_w4l:

  vbroadcasti128 m3, [srcq]
  vbroadcasti128 m4, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m3, m4, m8, m5, m6, xm13

  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m11, m12, m0, m1, m2, m3, m9, m10
  mova m2, m4

%ifidn %1, put
  RND_SHR_MIN_R m11, m12, m14, xm15, m7
%else
  RND_SHR_R m11, m12, m14, 6
%endif

  vextracti128 xm3, m11, 1
  vextracti128 xm4, m12, 1

  movd [dstq], xm11
  movd [dstq + 4], xm3
  movd [dstq + dsq], xm12
  movd [dstq + dsq + 4], xm4
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .hv_w4l

  RET

.hv_w48:
  lea ss3q, [ssq*3]

  mova m8, [spf_h_shuf]
  vpbroadcastd xm5, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8 + 2]
  vpmovsxbw m5, xm5

  lea myq, [jrq - .hv_jmp_tbl + subpel_filters + myq*8]
  vpbroadcastd xm9, [myq]
  vpbroadcastd xm10, [myq + 4]
  vpmovsxbw m9, xm9
  vpmovsxbw m10, xm10

  sub srcq, 2
  sub srcq, ss3q

  vbroadcasti128 m0, [srcq]
  vbroadcasti128 m1, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m0, m1, m8, m5, m6, xm13

  vbroadcasti128 m4, [srcq]
  vbroadcasti128 m3, [srcq + ssq]
  vbroadcasti128 m2, [srcq + ssq*2]
  add srcq, ss3q

  PUT_4TAP_HS2 m4, m3, m8, m5, m6, xm13
  PUT_4TAP_HS1 m2,     m8, m5, m6, xm13

  INTERLEAVE_REGS wd, m0, m1, m4, m3, m2
  punpckldq m0, m4
  punpckldq m1, m3

  vbroadcasti128 m3, [srcq]
  vbroadcasti128 m4, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m3, m4, m8, m5, m6, xm13

  INTERLEAVE_REGS wd, m2, m3, m4

.hv_w48l:

  vbroadcasti128 m11, [srcq]
  vbroadcasti128 m12, [srcq + ssq]
  lea srcq, [srcq + ssq*2]

  PUT_4TAP_HS2 m11, m12, m8, m5, m6, xm13

  INTERLEAVE_REGS wd, m4, m11, m12
  punpckldq m2, m4
  punpckldq m3, m11

  pmaddwd m11, m0, m9
  pmaddwd m4, m2, m10
  pmaddwd m12, m1, m9
  paddd m11, m4
  pmaddwd m4, m3, m10
  paddd m12, m4
  phaddd m11, m11
  phaddd m12, m12

%ifidn %1, put
  RND_SHR_MIN_R m11, m12, m14, xm15, m7
%else
  RND_SHR_R m11, m12, m14, 6
%endif

  vextracti128 xm4, m11, 1
  movd [dstq], xm11
  movd [dstq + 4], xm4
  vextracti128 xm4, m12, 1
  movd [dstq + dsq], xm12
  movd [dstq + dsq + 4], xm4
  lea dstq, [dstq + dsq*2]

  pshufd m0, m0, q2031
  pshufd m1, m1, q2031
  pshufd m11, m2, q3120
  pshufd m12, m3, q3120
  pshufd m2, m2, q2031
  pshufd m3, m3, q2031

  mova m4, m3
  psrad m4, 16
  packssdw m4, m4

  punpckldq m0, m11
  punpckldq m1, m12

  sub hd, 2
  jg .hv_w48l
  RET

.hv_w8:
  mov _wq, ss3q

  cmp hd, 4
  jg .hv_w88

  lea ss3q, [ssq*3]

  vpbroadcastq xm5, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8]
  vpmovsxbw m5, xm5

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, oh, tdst, tsrc, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, oh, tdst, tsrc, ds, ss3
%endif

  sub srcq, 6
  sub srcq, ssq

  mov ohd, hd
  mov tdstq, dstq
  mov tsrcq, srcq

.hv_w8l:

  PUT_8TAP_HS 0, 1, 2, m5, m6, xm13, 0*ssq
  PUT_8TAP_HS 1, 2, 3, m5, m6, xm13, 1*ssq
  PUT_8TAP_HS 2, 3, 4, m5, m6, xm13, 2*ssq
  add srcq, ss3q

  INTERLEAVE_REGS wd, m0, m1, m2

.hv_w8c: ; Nx2, Nx4

  PUT_8TAP_HS 3, 8, 11, m5, m6, xm13, 0*ssq
  PUT_8TAP_HS 4, 8, 11, m5, m6, xm13, 1*ssq
  lea srcq, [srcq + ssq*2]

  INTERLEAVE_REGS wd, m2, m3, m4

  MUL_ADD_R m8, m11, m0, m1, m2, m3, m9, m10
  mova m2, m4

%ifidn %1, put
  RND_SHR_MIN_R m8, m11, m14, xm15, m7
%else
  RND_SHR_R m8, m11, m14, 6
%endif

  vextracti128 xm3, m8, 1
  vextracti128 xm4, m11, 1

  movq [dstq], xm8
  movq [dstq + 8], xm3
  movq [dstq + dsq], xm11
  movq [dstq + dsq + 8], xm4
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .hv_w8c

  add tdstq, 2*8
  add tsrcq, 2*8
  mov hd, ohd
  mov dstq, tdstq
  mov srcq, tsrcq
  sub _wd, 8
  jg .hv_w8l
  RET

.hv_w88:
%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, mx, my, jr, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, mx, my, jr, ds, ss3
%endif

  lea ss3q, [ssq*3]

  vpbroadcastq xm7, [jrq - .hv_jmp_tbl + subpel_filters + mxq*8]
  vpmovsxbw m7, xm7

  sub srcq, 6
  sub srcq, ss3q

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, oh, tdst, bdmax, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, oh, tdst, bdmax, ds, ss3
%endif

  mov ohd, hd
  mov tdstq, dstq

  popcnt bdmaxd, bdmaxm
  cmp bdmaxd, 12
  je .hv_w88_12bit

%ifidn %1, put
  DEFINE_ARGS dst, ds, src, ss, _w, h, oh, tdst, tsrc, ss3
%elifidn %1, prep
  DEFINE_ARGS dst, src, ss, _w, h, oh, tdst, tsrc, ds, ss3
%endif

  mov tsrcq, srcq

.hv_w88l_10bit: ; Nx6, Nx8, Nx12, Nx16, Nx24, Nx32:

  vpbroadcastd m15, [pd_2]   ; (1 << (6 - 4) >> 1)

  PUT_8TAP_HS 0, 12, 13, m7, m15, 6 - 4, 0*ssq
  PUT_8TAP_HS 1, 12, 13, m7, m15, 6 - 4, 1*ssq
  PUT_8TAP_HS 2, 12, 13, m7, m15, 6 - 4, 2*ssq
  add srcq, ss3q

  PUT_8TAP_HS 3, 12, 13, m7, m15, 6 - 4, 0*ssq
  PUT_8TAP_HS 4, 12, 13, m7, m15, 6 - 4, 1*ssq
  lea srcq, [srcq + ssq*2]

  PUT_8TAP_HS 5, 12, 13, m7, m15, 6 - 4, 0*ssq
  PUT_8TAP_HS 6, 12, 13, m7, m15, 6 - 4, 1*ssq
  lea srcq, [srcq + ssq*2]

  INTERLEAVE_REGS wd, m0, m1, m2, m3, m4, m5, m6

.hv_w88c_10bit:

  PUT_8TAP_HS 12, 14, 15, m7, m15, 6 - 4, 0*ssq, [pd_2]
  PUT_8TAP_HS 13, 14, 15, m7, m15, 6 - 4, 1*ssq, [pd_2]
  lea srcq, [srcq + ssq*2]

  INTERLEAVE_REGS wd, m6, m12, m13

  MUL_ADD_R m14, m15, m0, m1, m2, m3, m8, m9
  MUL_ACC_R m14, m15, m2, m3, m4, m5, m10
  MUL_ACC_R m14, m15, m4, m5, m6, m12, m11

%ifidn %1, put
  vpbroadcastd m6, [pd_512]    ; (1 << (6 + 4) >> 1)
  vpbroadcastw m12, tsrcm      ; bdmaxm
  RND_SHR_MIN_R m14, m15, m6, 6 + 4, m12
%else
  vpbroadcastd m6, [nd_524256] ; (1 << 6 >> 1) - (8192 << 6)
  RND_SHR_R m14, m15, m6, 6
%endif

  mova m6, m13

  vextracti128 xm12, m14, 1
  vextracti128 xm13, m15, 1

  movq [dstq], xm14
  movq [dstq + 8], xm12
  movq [dstq + dsq], xm15
  movq [dstq + dsq + 8], xm13
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .hv_w88c_10bit

  add tdstq, 2*8
  add tsrcq, 2*8
  mov hd, ohd
  mov dstq, tdstq
  mov srcq, tsrcq
  sub _wd, 8
  jg .hv_w88l_10bit
  RET

.hv_w88_12bit:

  mov tsrcq, srcq

.hv_w88l_12bit: ; Nx6, Nx8, Nx12, Nx16, Nx24, Nx32:

  vpbroadcastd m15, [pd_8]   ; (1 << (6 - 2) >> 1)

  PUT_8TAP_HS 0, 12, 13, m7, m15, 6 - 2, 0*ssq
  PUT_8TAP_HS 1, 12, 13, m7, m15, 6 - 2, 1*ssq
  PUT_8TAP_HS 2, 12, 13, m7, m15, 6 - 2, 2*ssq
  add srcq, ss3q

  PUT_8TAP_HS 3, 12, 13, m7, m15, 6 - 2, 0*ssq
  PUT_8TAP_HS 4, 12, 13, m7, m15, 6 - 2, 1*ssq
  lea srcq, [srcq + ssq*2]

  PUT_8TAP_HS 5, 12, 13, m7, m15, 6 - 2, 0*ssq
  PUT_8TAP_HS 6, 12, 13, m7, m15, 6 - 2, 1*ssq
  lea srcq, [srcq + ssq*2]

  INTERLEAVE_REGS wd, m0, m1, m2, m3, m4, m5, m6

.hv_w88c_12bit:

  PUT_8TAP_HS 12, 14, 15, m7, m15, 6 - 2, 0*ssq, [pd_8]
  PUT_8TAP_HS 13, 14, 15, m7, m15, 6 - 2, 1*ssq, [pd_8]
  lea srcq, [srcq + ssq*2]

  INTERLEAVE_REGS wd, m6, m12, m13

  MUL_ADD_R m14, m15, m0, m1, m2, m3, m8, m9
  MUL_ACC_R m14, m15, m2, m3, m4, m5, m10
  MUL_ACC_R m14, m15, m4, m5, m6, m12, m11

%ifidn %1, put
  vpbroadcastd m6, [pd_128]    ; (1 << (6 + 2) >> 1)
  vpbroadcastw m12, tsrcm      ; bdmaxm
  RND_SHR_MIN_R m14, m15, m6, 6 + 2, m12
%else
  vpbroadcastd m6, [nd_524256] ; (1 << 6 >> 1) - (8192 << 6)
  RND_SHR_R m14, m15, m6, 6
%endif

  mova m6, m13

  vextracti128 xm12, m14, 1
  vextracti128 xm13, m15, 1

  movq [dstq], xm14
  movq [dstq + 8], xm12
  movq [dstq + dsq], xm15
  movq [dstq + dsq + 8], xm13
  lea dstq, [dstq + dsq*2]

  sub hd, 2
  jg .hv_w88c_12bit

  add tdstq, 2*8
  add tsrcq, 2*8
  mov hd, ohd
  mov dstq, tdstq
  mov srcq, tsrcq
  sub _wd, 8
  jg .hv_w88l_12bit
  RET

.hv_jmp_tbl:
%ifidn %1, put
  dd .hv_w2 - .hv_jmp_tbl
%endif
  dd .hv_w4 - .hv_jmp_tbl
  dd .hv_w8 - .hv_jmp_tbl
  dd .hv_w8 - .hv_jmp_tbl
  dd .hv_w8 - .hv_jmp_tbl
  dd .hv_w8 - .hv_jmp_tbl
  dd .hv_w8 - .hv_jmp_tbl
%endm

filter_fn put
filter_fn prep

%endif ; ARCH_X86_64
