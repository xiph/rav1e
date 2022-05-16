; Copyright (c) 2022, The rav1e contributors. All rights reserved
;
; This source code is subject to the terms of the BSD 2 Clause License and
; the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
; was not distributed with this source code in the LICENSE file, you can
; obtain it at www.aomedia.org/license/software. If the Alliance for Open
; Media Patent License 1.0 was not distributed with this source code in the
; PATENTS file, you can obtain it at www.aomedia.org/license/patent.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION_RODATA

align 32
mask_lut: db \
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, \
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,

SECTION .text

%if ARCH_X86_64

%macro SAD_ROW_FN 0
cglobal sad_row_8bpc, 3, 5, 8, p1, p2, len, \
                      resid_simd, resid
  ; always need to zero first register
  pxor    xm0, xm0
  mov     resid_simdq, lenq
  mov     residq,  lenq
  and     residd, mmsize - 1
  and     resid_simdq, -(mmsize)
  jz     .residual
  ; need to zero 2 more registers
  pxor    xm1, xm1
  pxor    xm2, xm2
  ; len = num elements processed by unrolled loop
  and     lenq, -(4*mmsize)
  jz     .lt_4x
  add     p1q, lenq
  add     p2q, lenq
  sub     resid_simdq, lenq
  ; last register needed for unrolled loop
  pxor    xm3, xm3
  neg     lenq
.loop:
  mova        m4,     [p1q + lenq + 0*mmsize]
  mova        m5,     [p1q + lenq + 1*mmsize]
  mova        m6,     [p1q + lenq + 2*mmsize]
  mova        m7,     [p1q + lenq + 3*mmsize]

  psadbw      m4, m4, [p2q + lenq + 0*mmsize]
  psadbw      m5, m5, [p2q + lenq + 1*mmsize]
  psadbw      m6, m6, [p2q + lenq + 2*mmsize]
  psadbw      m7, m7, [p2q + lenq + 3*mmsize]

  paddq       m0, m4
  paddq       m1, m5
  paddq       m2, m6
  paddq       m3, m7

  add         lenq, 4*mmsize
  jnz        .loop

  paddq       m2, m3
.lt_4x:
  ; jump to correct place for residual vector reduction
  test      resid_simdd, resid_simdd
  jz       .residual_setup
  cmp       resid_simdd, 1*mmsize
  je       .r1
  cmp       resid_simdd, 2*mmsize
  je       .r2
  ; add residual vectors in reverse order
  mova      m6,     [p1q + lenq + 2*mmsize]
  psadbw    m6, m6, [p2q + lenq + 2*mmsize]
  paddq     m2, m6
.r2:
  mova      m5,     [p1q + lenq + 1*mmsize]
  psadbw    m5, m5, [p2q + lenq + 1*mmsize]
  paddq     m1, m5
.r1:
  mova      m4,     [p1q + lenq + 0*mmsize]
  psadbw    m4, m4, [p2q + lenq + 0*mmsize]
  paddq     m0, m4
.residual_setup:
  paddq     m1, m2
  paddq     m0, m1
  test  residd, residd
  jz       .hsum
.residual:
  DEFINE_ARGS p1, p2, mask_ptr, resid_simd, resid
  lea   mask_ptrq, [mask_lut]
  shl      residd, 5
  mova     m1,     [mask_ptrq + residq]
  ; read residual elements and zero out elements
  ; past the end of the array with mask in m1
  pand     m2, m1, [p1q + resid_simdq]
  pand     m3, m1, [p2q + resid_simdq]
  psadbw   m1, m2, m3
  paddq    m0, m1
.hsum:
%if mmsize == 32
  vextracti128  xm1, ym0, 1
  paddq         xm0, xm1
%endif
  pshufd        xm1, xm0, q0032
  paddq         xm0, xm1
  movq          rax, xm0
  RET
%endmacro

INIT_XMM sse2
SAD_ROW_FN

INIT_YMM avx2
SAD_ROW_FN

%endif ; ARCH_X86_64
