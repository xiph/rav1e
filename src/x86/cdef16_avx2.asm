; Copyright (c) 2017-2021, The rav1e contributors
; Copyright (c) 2021, Nathan Egge
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

SECTION .text

cextern cdef_dir_8bpc_avx2

INIT_YMM avx2
cglobal cdef_dir_16bpc, 4, 4, 3, 32 + 8*8, src, ss, var, bdmax
  popcnt bdmaxd, bdmaxd
  movzx bdmaxq, bdmaxw
  sub bdmaxq, 8
  movq xm2, bdmaxq
  DEFINE_ARGS src, ss, var, ss3
  lea ss3q, [ssq*3]
  mova xm0, [srcq + ssq*0]
  mova xm1, [srcq + ssq*1]
  vinserti128 m0, [srcq + ssq*2], 1
  vinserti128 m1, [srcq + ss3q], 1
  psraw m0, xm2
  psraw m1, xm2
  vpackuswb m0, m1
  mova [rsp + 32 + 0*8], m0
  lea srcq, [srcq + ssq*4]
  mova xm0, [srcq + ssq*0]
  mova xm1, [srcq + ssq*1]
  vinserti128 m0, [srcq + ssq*2], 1
  vinserti128 m1, [srcq + ss3q], 1
  psraw m0, xm2
  psraw m1, xm2
  vpackuswb m0, m1
  mova [rsp + 32 + 4*8], m0
  lea srcq, [rsp + 32] ; WIN64 shadow space
  mov ssq, 8
  call mangle(private_prefix %+ _cdef_dir_8bpc %+ SUFFIX)
  RET

%endif ; ARCH_X86_64
