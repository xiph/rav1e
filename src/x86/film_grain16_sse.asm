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

SECTION_RODATA 16
pd_16: times 4 dd 16
pw_1: times 8 dw 1
pw_8192: times 8 dw 8192
pw_23_22: times 4 dw 23, 22
pb_mask: db 0, 0x80, 0x80, 0, 0x80, 0, 0, 0x80, 0x80, 0, 0, 0x80, 0, 0x80, 0x80, 0
rnd_next_upperbit_mask: dw 0x100B, 0x2016, 0x402C, 0x8058
pw_seed_xor: times 2 dw 0xb524
             times 2 dw 0x49d8
pb_1: times 4 db 1
hmul_bits: dw 32768, 16384, 8192, 4096
round: dw 2048, 1024, 512
mul_bits: dw 256, 128, 64, 32, 16
round_vals: dw 32, 64, 128, 256, 512, 1024
max: dw 256*4-1, 240*4, 235*4, 256*16-1, 240*16, 235*16
min: dw 0, 16*4, 16*16
pw_27_17_17_27: dw 27, 17, 17, 27
; these two should be next to each other
pw_4: times 2 dw 4
pw_16: times 2 dw 16

%macro JMP_TABLE 1-*
    %xdefine %1_table %%table
    %xdefine %%base %1_table
    %xdefine %%prefix mangle(private_prefix %+ _%1)
    %%table:
    %rep %0 - 1
        dd %%prefix %+ .ar%2 - %%base
        %rotate 1
    %endrep
%endmacro

JMP_TABLE generate_grain_y_16bpc_ssse3, 0, 1, 2, 3
JMP_TABLE generate_grain_uv_420_16bpc_ssse3, 0, 1, 2, 3

struc FGData
    .seed:                      resd 1
    .num_y_points:              resd 1
    .y_points:                  resb 14 * 2
    .chroma_scaling_from_luma:  resd 1
    .num_uv_points:             resd 2
    .uv_points:                 resb 2 * 10 * 2
    .scaling_shift:             resd 1
    .ar_coeff_lag:              resd 1
    .ar_coeffs_y:               resb 24
    .ar_coeffs_uv:              resb 2 * 28 ; includes padding
    .ar_coeff_shift:            resq 1
    .grain_scale_shift:         resd 1
    .uv_mult:                   resd 2
    .uv_luma_mult:              resd 2
    .uv_offset:                 resd 2
    .overlap_flag:              resd 1
    .clip_to_restricted_range:  resd 1
endstruc

cextern gaussian_sequence

SECTION .text

%macro REPX 2-*
    %xdefine %%f(x) %1
%rep %0 - 1
    %rotate 1
    %%f(%1)
%endrep
%endmacro

%define m(x) mangle(private_prefix %+ _ %+ x %+ SUFFIX)

%macro vpgatherdw 5-8 8, 1 ; dst, src, base, tmp_gpr[x2], cnt, stride, tmp_xmm_reg
%assign %%idx 0
%define %%tmp %2
%if %0 == 8
%define %%tmp %8
%endif
%rep (%6/2)
%if %%idx == 0
    movd        %5 %+ d, %2
    pshuflw       %%tmp, %2, q3232
%else
    movd        %5 %+ d, %%tmp
%if %6 == 8
%if %%idx == 2
    punpckhqdq    %%tmp, %%tmp
%elif %%idx == 4
    psrlq         %%tmp, 32
%endif
%endif
%endif
    movzx       %4 %+ d, %5 %+ w
    shr         %5 %+ d, 16

%if %%idx == 0
    movd             %1, [%3+%4*%7]
%else
    pinsrw           %1, [%3+%4*%7], %%idx + 0
%endif
    pinsrw           %1, [%3+%5*%7], %%idx + 1
%assign %%idx %%idx+2
%endrep
%endmacro

%macro SPLATD 2 ; dst, src
%ifnidn %1, %2
    movd %1, %2
%endif
    pshufd %1, %1, q0000
%endmacro

%macro SPLATW 2 ; dst, src
%ifnidn %1, %2
    movd %1, %2
%endif
    pshuflw %1, %1, q0000
    punpcklqdq %1, %1
%endmacro


INIT_XMM ssse3
cglobal generate_grain_y_16bpc, 3, 9, 16, buf, fg_data, bdmax
    lea              r4, [pb_mask]
%define base r4-pb_mask
    movq             m1, [base+rnd_next_upperbit_mask]
    movq             m4, [base+mul_bits]
    movq             m7, [base+hmul_bits]
    mov             r3d, [fg_dataq+FGData.grain_scale_shift]
    lea             r6d, [bdmaxq+1]
    shr             r6d, 11             ; 0 for 10bpc, 2 for 12bpc
    sub              r3, r6
    SPLATW           m8, [base+round+r3*2-2]
    mova             m5, [base+pb_mask]
    SPLATW           m0, [fg_dataq+FGData.seed]
    mov              r3, -73*82*2
    sub            bufq, r3
    lea              r6, [gaussian_sequence]
.loop:
    pand             m2, m0, m1
    psrlw            m3, m2, 10
    por              m2, m3             ; bits 0xf, 0x1e, 0x3c and 0x78 are set
    pmullw           m2, m4             ; bits 0x0f00 are set
    pshufb           m6, m5, m2         ; set 15th bit for next 4 seeds
    psllq            m2, m6, 30
    por              m2, m6
    psllq            m6, m2, 15
    por              m2, m6             ; aggregate each bit into next seed's high bit
    pmulhuw          m3, m0, m7
    por              m2, m3             ; 4 next output seeds
    pshuflw          m0, m2, q3333
    psrlw            m2, 5
    vpgatherdw       m3, m2, r6, r5, r7, 4, 2
    paddw            m3, m3             ; otherwise bpc=12 w/ grain_scale_shift=0
                                        ; shifts by 0, which pmulhrsw does not support
    pmulhrsw         m3, m8
    movq      [bufq+r3], m3
    add              r3, 4*2
    jl .loop

    ; auto-regression code
    movsxd           r3, [fg_dataq+FGData.ar_coeff_lag]
    movsxd           r3, [base+generate_grain_y_16bpc_ssse3_table+r3*4]
    lea              r3, [r3+base+generate_grain_y_16bpc_ssse3_table]
    jmp              r3

.ar1:
%if WIN64
    DEFINE_ARGS shift, fg_data, max, buf, val3, min, cf3, x, val0
    lea            bufq, [r0-2*(82*73-(82*3+79))]
%elif ARCH_X86_64
    DEFINE_ARGS buf, fg_data, max, shift, val3, min, cf3, x, val0
    sub            bufq, 2*(82*73-(82*3+79))
%else
    ; FIXME shift goes into r1 (x86-32 code)
    ..
%endif
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    movsx          cf3d, byte [fg_dataq+FGData.ar_coeffs_y+3]
    movd             m4, [fg_dataq+FGData.ar_coeffs_y]
%if WIN64
    DEFINE_ARGS shift, h, max, buf, val3, min, cf3, x, val0
%elif ARCH_X86_64
    DEFINE_ARGS buf, h, max, shift, val3, min, cf3, x, val0
%else
    ; x86-32 code
    ..
%endif
%if cpuflag(sse4)
    pmovsxbw         m4, m4
%else
    pxor             m3, m3
    pcmpgtb          m3, m4
    punpcklbw        m4, m3
%endif
    pinsrw           m4, [pw_1], 3
    pshufd           m5, m4, q1111
    pshufd           m4, m4, q0000
    SPLATW           m3, [base+round_vals+shiftq*2-12]    ; rnd
    mov              hd, 70
    sar            maxd, 1
    mov            mind, maxd
    xor            mind, -1
.y_loop_ar1:
    mov              xq, -76
    movsx         val3d, word [bufq+xq*2-2]
.x_loop_ar1:
    movu             m0, [bufq+xq*2-82*2-2]     ; top/left
    psrldq           m2, m0, 2                  ; top
    psrldq           m1, m0, 4                  ; top/right
    punpcklwd        m0, m2
    punpcklwd        m1, m3
    pmaddwd          m0, m4
    pmaddwd          m1, m5
    paddd            m0, m1
.x_loop_ar1_inner:
    movd          val0d, m0
    psrldq           m0, 4
    imul          val3d, cf3d
    add           val3d, val0d
    sar           val3d, shiftb
    movsx         val0d, word [bufq+xq*2]
    add           val3d, val0d
    cmp           val3d, maxd
    cmovg         val3d, maxd
    cmp           val3d, mind
    cmovl         val3d, mind
    mov word [bufq+xq*2], val3w
    ; keep val3d in-place as left for next x iteration
    inc              xq
    jz .x_loop_ar1_end
    test             xq, 3
    jnz .x_loop_ar1_inner
    jmp .x_loop_ar1

.x_loop_ar1_end:
    add            bufq, 82*2
    dec              hd
    jg .y_loop_ar1
.ar0:
    RET

.ar2:
    DEFINE_ARGS buf, fg_data, bdmax, shift
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    SPLATW          m12, [base+round_vals-12+shiftq*2]
    movu             m6, [fg_dataq+FGData.ar_coeffs_y+0]    ; cf0-11
    pxor             m9, m9
    punpcklwd       m12, m9
    pcmpgtb          m9, m6
    punpckhbw       m10, m6, m9
    punpcklbw        m6, m9
    pshufd           m9, m6, q3333
    pshufd           m8, m6, q2222
    pshufd           m7, m6, q1111
    pshufd           m6, m6, q0000
    pshufd          m11, m10, q1111
    pshufd          m10, m10, q0000
    sar          bdmaxd, 1
    SPLATW          m13, bdmaxd                             ; max_grain
    pcmpeqw         m14, m14
%if !cpuflag(sse4)
    pcmpeqw         m15, m15
    psrldq          m15, 14
    pslldq          m15, 2
    pxor            m15, m14
%endif
    pxor            m14, m13                                ; min_grain
    sub            bufq, 2*(82*73-(82*3+79))
    DEFINE_ARGS buf, fg_data, h, x
    mov              hd, 70
.y_loop_ar2:
    mov              xq, -76

.x_loop_ar2:
    movu             m0, [bufq+xq*2-82*4-4]     ; y=-2,x=[-2,+5]
    movu             m1, [bufq+xq*2-82*2-4]     ; y=-1,x=[-2,+5]
    psrldq           m2, m0, 2
    psrldq           m3, m0, 4
    psrldq           m4, m0, 6
    psrldq           m5, m0, 8
    punpcklwd        m0, m2
    punpcklwd        m3, m4
    punpcklwd        m5, m1
    psrldq           m2, m1, 2
    psrldq           m4, m1, 4
    punpcklwd        m2, m4
    psrldq           m4, m1, 6
    psrldq           m1, 8
    punpcklwd        m4, m1
    pmaddwd          m0, m6
    pmaddwd          m3, m7
    pmaddwd          m5, m8
    pmaddwd          m2, m9
    pmaddwd          m4, m10
    paddd            m0, m3
    paddd            m5, m2
    paddd            m0, m4
    paddd            m0, m5                     ; accumulated top 2 rows
    paddd            m0, m12

    movu             m1, [bufq+xq*2-4]      ; y=0,x=[-2,+5]
    pshufd           m4, m1, q3321
    pxor             m2, m2
    pcmpgtw          m2, m4
    punpcklwd        m4, m2                 ; in dwords, y=0,x=[0,3]
.x_loop_ar2_inner:
    pmaddwd          m2, m1, m11
    paddd            m2, m0
    psrldq           m0, 4                  ; shift top to next pixel
    psrad            m2, [fg_dataq+FGData.ar_coeff_shift]
    paddd            m2, m4
    packssdw         m2, m2
    pminsw           m2, m13
    pmaxsw           m2, m14
    psrldq           m4, 4
    pslldq           m2, 2
    psrldq           m1, 2
%if cpuflag(sse4)
    pblendw          m1, m2, 00000010b
%else
    pand             m1, m15
    pandn            m3, m15, m2
    por              m1, m3
%endif
    ; overwrite previous pixel, this should be ok
    movd  [bufq+xq*2-2], m1
    inc              xq
    jz .x_loop_ar2_end
    test             xq, 3
    jnz .x_loop_ar2_inner
    jmp .x_loop_ar2

.x_loop_ar2_end:
    add            bufq, 82*2
    dec              hd
    jg .y_loop_ar2
    RET

.ar3:
    DEFINE_ARGS buf, fg_data, bdmax, shift
%if WIN64
    mov              r6, rsp
    and             rsp, ~15
    sub             rsp, 64
    %define         tmp  rsp
%else
    %define         tmp  rsp+stack_offset-72
%endif
    sar          bdmaxd, 1
    SPLATW          m15, bdmaxd                                 ; max_grain
    pcmpeqw         m14, m14
%if !cpuflag(sse4)
    pcmpeqw         m12, m12
    psrldq          m12, 14
    pslldq          m12, 4
    pxor            m12, m14
%endif
    pxor            m14, m15                                   ; min_grain
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]

    ; build cf0-1 until 18-19 in m5-12 and r0/1
    pxor             m1, m1
    movu             m0, [fg_dataq+FGData.ar_coeffs_y+ 0]       ; cf0-15
    pcmpgtb          m1, m0
    punpckhbw        m2, m0, m1
    punpcklbw        m0, m1

%if cpuflag(sse4)
    pshufd          m12, m2, q3333
%else
    pshufd          m13, m2, q3333
    mova       [tmp+48], m13
%endif
    pshufd          m11, m2, q2222
    pshufd          m10, m2, q1111
    pshufd           m9, m2, q0000
    pshufd           m8, m0, q3333
    pshufd           m7, m0, q2222
    pshufd           m6, m0, q1111
    pshufd           m5, m0, q0000

    ; build cf20,round in r2
    ; build cf21-23,round*2 in m13
    pxor             m1, m1
    movq             m0, [fg_dataq+FGData.ar_coeffs_y+16]       ; cf16-23
    pcmpgtb          m1, m0
    punpcklbw        m0, m1
    pshufd           m1, m0, q0000
    pshufd           m2, m0, q1111
    mova       [tmp+ 0], m1
    mova       [tmp+16], m2
    psrldq          m13, m0, 10
    pinsrw          m13, [base+round_vals+shiftq*2-10], 3
    pinsrw           m0, [base+round_vals+shiftq*2-12], 5
    pshufd           m3, m0, q2222
    mova       [tmp+32], m3

    DEFINE_ARGS buf, fg_data, h, x
    sub            bufq, 2*(82*73-(82*3+79))
    mov              hd, 70
.y_loop_ar3:
    mov              xq, -76

.x_loop_ar3:
    movu             m0, [bufq+xq*2-82*6-6+ 0]      ; y=-3,x=[-3,+4]
    movd             m1, [bufq+xq*2-82*6-6+16]      ; y=-3,x=[+5,+6]
    palignr          m2, m1, m0, 2                  ; y=-3,x=[-2,+5]
    palignr          m1, m1, m0, 12                 ; y=-3,x=[+3,+6]
    punpckhwd        m3, m0, m2                     ; y=-3,x=[+1/+2,+2/+3,+3/+4,+4/+5]
    punpcklwd        m0, m2                         ; y=-3,x=[-3/-2,-2/-1,-1/+0,+0/+1]
    shufps           m2, m0, m3, q1032              ; y=-3,x=[-1/+0,+0/+1,+1/+2,+2/+3]

    pmaddwd          m0, m5
    pmaddwd          m2, m6
    pmaddwd          m3, m7
    paddd            m0, m2
    paddd            m0, m3
    ; m0 = top line first 6 multiplied by cf, m1 = top line last entry

    movu             m2, [bufq+xq*2-82*4-6+ 0]      ; y=-2,x=[-3,+4]
    movd             m3, [bufq+xq*2-82*4-6+16]      ; y=-2,x=[+5,+6]
    punpcklwd        m1, m2                         ; y=-3/-2,x=[+3/-3,+4/-2,+5/-1,+6/+0]
    palignr          m4, m3, m2, 2                  ; y=-3,x=[-2,+5]
    palignr          m3, m3, m2, 4                  ; y=-3,x=[-1,+6]
    punpckhwd        m2, m4, m3                     ; y=-2,x=[+2/+3,+3/+4,+4/+5,+5/+6]
    punpcklwd        m4, m3                         ; y=-2,x=[-2/-1,-1/+0,+0/+1,+1/+2]
    shufps           m3, m4, m2, q1032              ; y=-2,x=[+0/+1,+1/+2,+2/+3,+3/+4]

    pmaddwd          m1, m8
    pmaddwd          m4, m9
    pmaddwd          m3, m10
    pmaddwd          m2, m11
    paddd            m1, m4
    paddd            m3, m2
    paddd            m0, m1
    paddd            m0, m3
    ; m0 = top 2 lines multiplied by cf

    movu             m1, [bufq+xq*2-82*2-6+ 0]      ; y=-1,x=[-3,+4]
    movd             m2, [bufq+xq*2-82*2-6+16]      ; y=-1,x=[+5,+6]
    palignr          m3, m2, m1, 2                  ; y=-1,x=[-2,+5]
    palignr          m2, m2, m1, 12                 ; y=-1,x=[+3,+6]
    punpckhwd        m4, m1, m3                     ; y=-1,x=[+1/+2,+2/+3,+3/+4,+4/+5]
    punpcklwd        m1, m3                         ; y=-1,x=[-3/-2,-2/-1,-1/+0,+0/+1]
    shufps           m3, m1, m4, q1032              ; y=-1,x=[-1/+0,+0/+1,+1/+2,+2/+3]
    punpcklwd        m2, [pw_1]

%if cpuflag(sse4)
    pmaddwd          m1, m12
%else
    pmaddwd          m1, [tmp+48]
%endif
    pmaddwd          m3, [tmp+ 0]
    pmaddwd          m4, [tmp+16]
    pmaddwd          m2, [tmp+32]
    paddd            m1, m3
    paddd            m4, m2
    paddd            m0, m1
    paddd            m0, m4
    ; m0 = top 3 lines multiplied by cf plus rounding for downshift

    movu             m1, [bufq+xq*2-6]      ; y=0,x=[-3,+4]
.x_loop_ar3_inner:
    pmaddwd          m2, m1, m13
    pshufd           m3, m2, q1111
    paddd            m2, m3                 ; left+cur
    paddd            m2, m0                 ; add top
    psrldq           m0, 4
    psrad            m2, [fg_dataq+FGData.ar_coeff_shift]
    packssdw         m2, m2
    pminsw           m2, m15
    pmaxsw           m2, m14
    pslldq           m2, 4
    psrldq           m1, 2
%if cpuflag(sse4)
    pblendw          m1, m2, 00000100b
%else
    pand             m1, m12
    pandn            m3, m12, m2
    por              m1, m3
%endif
    ; overwrite a couple of pixels, should be ok
    movq  [bufq+xq*2-4], m1
    inc              xq
    jz .x_loop_ar3_end
    test             xq, 3
    jnz .x_loop_ar3_inner
    jmp .x_loop_ar3

.x_loop_ar3_end:
    add            bufq, 82*2
    dec              hd
    jg .y_loop_ar3
%if WIN64
    mov             rsp, r6
%endif
    RET

INIT_XMM ssse3
cglobal generate_grain_uv_420_16bpc, 4, 11, 16, buf, bufy, fg_data, uv, bdmax
%define base r8-pb_mask
    lea              r8, [pb_mask]
    movifnidn    bdmaxd, bdmaxm
    movq             m1, [base+rnd_next_upperbit_mask]
    movq             m4, [base+mul_bits]
    movq             m7, [base+hmul_bits]
    mov             r5d, [fg_dataq+FGData.grain_scale_shift]
    lea             r6d, [bdmaxq+1]
    shr             r6d, 11             ; 0 for 10bpc, 2 for 12bpc
    sub              r5, r6
    SPLATW           m8, [base+round+r5*2-2]
    mova             m5, [base+pb_mask]
    SPLATW           m0, [fg_dataq+FGData.seed]
    SPLATW           m9, [base+pw_seed_xor+uvq*4]
    pxor             m0, m9
    lea              r6, [gaussian_sequence]
    mov             r7d, 38
    add            bufq, 44*2
.loop_y:
    mov              r5, -44
.loop_x:
    pand             m2, m0, m1
    psrlw            m3, m2, 10
    por              m2, m3             ; bits 0xf, 0x1e, 0x3c and 0x78 are set
    pmullw           m2, m4             ; bits 0x0f00 are set
    pshufb           m6, m5, m2         ; set 15th bit for next 4 seeds
    psllq            m2, m6, 30
    por              m2, m6
    psllq            m6, m2, 15
    por              m2, m6             ; aggregate each bit into next seed's high bit
    pmulhuw          m3, m0, m7
    por              m2, m3             ; 4 next output seeds
    pshuflw          m0, m2, q3333
    psrlw            m2, 5
    vpgatherdw       m3, m2, r6, r9, r10, 4, 2
    paddw            m3, m3             ; otherwise bpc=12 w/ grain_scale_shift=0
                                        ; shifts by 0, which pmulhrsw does not support
    pmulhrsw         m3, m8
    movq    [bufq+r5*2], m3
    add              r5, 4
    jl .loop_x
    add            bufq, 82*2
    dec             r7d
    jg .loop_y

    ; auto-regression code
    movsxd           r5, [fg_dataq+FGData.ar_coeff_lag]
    movsxd           r5, [base+generate_grain_uv_420_16bpc_ssse3_table+r5*4]
    lea              r5, [r5+base+generate_grain_uv_420_16bpc_ssse3_table]
    jmp              r5

.ar0:
    DEFINE_ARGS buf, bufy, fg_data, uv, bdmax, shift
    imul            uvd, 28
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    movd             m4, [fg_dataq+FGData.ar_coeffs_uv+uvq]
    SPLATW           m3, [base+hmul_bits+shiftq*2-10]
    sar          bdmaxd, 1
    SPLATW          m14, bdmaxd                     ; max_gain
    pcmpeqw          m7, m7
    pxor             m7, m14                        ; min_grain
    DEFINE_ARGS buf, bufy, h, x
    pxor             m5, m5
    pcmpgtb          m5, m4
    punpcklbw        m4, m5
    SPLATW           m6, [hmul_bits+4]
    SPLATW           m4, m4
    pxor             m5, m5
%if !cpuflag(sse4)
    pcmpeqw         m12, m12
    pslldq          m12, 12
%endif
    sub            bufq, 2*(82*38+82-(82*3+41))
    add           bufyq, 2*(3+82*3)
    mov              hd, 35
.y_loop_ar0:
    ; first 32 pixels
    xor              xd, xd
.x_loop_ar0:
    movu             m8, [bufyq+xq*4]
    movu             m9, [bufyq+xq*4+82*2]
    movu            m10, [bufyq+xq*4     +16]
    movu            m11, [bufyq+xq*4+82*2+16]
    paddw            m8, m9
    paddw           m10, m11
    phaddw           m8, m10
    pmulhrsw         m8, m6
    punpckhwd        m9, m8, m5
    punpcklwd        m8, m5
    REPX {pmaddwd x, m4}, m8, m9
    REPX {psrad x, 5}, m8, m9
    packssdw         m8, m9
    pmulhrsw         m8, m3
    movu             m0, [bufq+xq*2]
    paddw            m8, m0
    pminsw           m8, m14
    pmaxsw           m8, m7
    cmp              xd, 32
    je .end
    movu    [bufq+xq*2], m8
    add              xd, 8
    jmp .x_loop_ar0

    ; last 6 pixels
.end:
%if cpuflag(sse4)
    pblendw          m8, m0, 11000000b
%else
    pand             m0, m12
    pandn            m9, m12, m8
    por              m8, m0, m9
%endif
    movu    [bufq+xq*2], m8

    add            bufq, 82*2
    add           bufyq, 82*4
    dec              hd
    jg .y_loop_ar0
    RET

.ar1:
    DEFINE_ARGS buf, bufy, fg_data, uv, max, cf3, min, val3, x
    imul            uvd, 28
    movsx          cf3d, byte [fg_dataq+FGData.ar_coeffs_uv+uvq+3]
    movq             m4, [fg_dataq+FGData.ar_coeffs_uv+uvq]
%if WIN64
    DEFINE_ARGS shift, bufy, h, buf, max, cf3, min, val3, x, val0
    lea            bufq, [r0-2*(82*38+44-(82*3+41))]
%elif ARCH_X86_64
    DEFINE_ARGS buf, bufy, h, shift, max, cf3, min, val3, x, val0
    sub            bufq, 2*(82*38+44-(82*3+41))
%else
    ; x86-32 code - move shift into r1 [ecx]
    ..
%endif
    mov          shiftd, [r2+FGData.ar_coeff_shift]
    pxor             m5, m5
    pcmpgtb          m5, m4
    punpcklbw        m4, m5                 ; cf0-4 in words
    pshuflw          m4, m4, q2100
    psrldq           m4, 2                  ; cf0-3,4 in words
    pshufd           m5, m4, q1111
    pshufd           m4, m4, q0000
    movd             m3, [base+round_vals+shiftq*2-12]    ; rnd
    pxor             m6, m6
    punpcklwd        m3, m6
    SPLATW           m6, [hmul_bits+4]
    SPLATD           m3, m3
    add           bufyq, 2*(79+82*3)
    mov              hd, 35
    sar            maxd, 1
    mov            mind, maxd
    xor            mind, -1
.y_loop_ar1:
    mov              xq, -38
    movsx         val3d, word [bufq+xq*2-2]
.x_loop_ar1:
    movu             m0, [bufq+xq*2-82*2-2] ; top/left
    movu             m8, [bufyq+xq*4]
    movu             m9, [bufyq+xq*4+82*2]
    psrldq           m2, m0, 2              ; top
    psrldq           m1, m0, 4              ; top/right
    phaddw           m8, m9
    pshufd           m9, m8, q3232
    paddw            m8, m9
    pmulhrsw         m8, m6
    punpcklwd        m0, m2
    punpcklwd        m1, m8
    pmaddwd          m0, m4
    pmaddwd          m1, m5
    paddd            m0, m1
    paddd            m0, m3
.x_loop_ar1_inner:
    movd          val0d, m0
    psrldq           m0, 4
    imul          val3d, cf3d
    add           val3d, val0d
    sar           val3d, shiftb
    movsx         val0d, word [bufq+xq*2]
    add           val3d, val0d
    cmp           val3d, maxd
    cmovg         val3d, maxd
    cmp           val3d, mind
    cmovl         val3d, mind
    mov word [bufq+xq*2], val3w
    ; keep val3d in-place as left for next x iteration
    inc              xq
    jz .x_loop_ar1_end
    test             xq, 3
    jnz .x_loop_ar1_inner
    jmp .x_loop_ar1

.x_loop_ar1_end:
    add            bufq, 82*2
    add           bufyq, 82*4
    dec              hd
    jg .y_loop_ar1
    RET

.ar2:
    DEFINE_ARGS buf, bufy, fg_data, uv, bdmax, shift
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    imul            uvd, 28
    sar          bdmaxd, 1
    SPLATW          m13, bdmaxd                 ; max_grain
    pcmpeqw         m14, m14
%if !cpuflag(sse4)
    pcmpeqw         m15, m15
    psrldq          m15, 14
    pslldq          m15, 2
    pxor            m15, m14
%endif
    pxor            m14, m13                    ; min_grain
%if cpuflag(sse4)
    SPLATW          m15, [hmul_bits+4]
%endif

    ; coef values
    movu             m0, [fg_dataq+FGData.ar_coeffs_uv+uvq+0]
    pxor             m1, m1
    pcmpgtb          m1, m0
    punpckhbw        m2, m0, m1
    punpcklbw        m0, m1
    pinsrw           m2, [base+round_vals-12+shiftq*2], 5

    pshufd           m6, m0, q0000
    pshufd           m7, m0, q1111
    pshufd           m8, m0, q2222
    pshufd           m9, m0, q3333
    pshufd          m10, m2, q0000
    pshufd          m11, m2, q1111
    pshufd          m12, m2, q2222

    DEFINE_ARGS buf, bufy, fg_data, h, x
    sub            bufq, 2*(82*38+44-(82*3+41))
    add           bufyq, 2*(79+82*3)
    mov              hd, 35
.y_loop_ar2:
    mov              xq, -38

.x_loop_ar2:
    movu             m0, [bufq+xq*2-82*4-4]     ; y=-2,x=[-2,+5]
    movu             m5, [bufq+xq*2-82*2-4]     ; y=-1,x=[-2,+5]
    psrldq           m4, m0, 2                  ; y=-2,x=[-1,+5]
    psrldq           m1, m0, 4                  ; y=-2,x=[-0,+5]
    psrldq           m3, m0, 6                  ; y=-2,x=[+1,+5]
    psrldq           m2, m0, 8                  ; y=-2,x=[+2,+5]
    punpcklwd        m0, m4                     ; y=-2,x=[-2/-1,-1/+0,+0/+1,+1/+2]
    punpcklwd        m1, m3                     ; y=-2,x=[+0/+1,+1/+2,+2/+3,+3/+4]
    punpcklwd        m2, m5                     ; y=-2/-1,x=[+2/-2,+3/-1,+4/+0,+5/+1]
    pmaddwd          m0, m6
    pmaddwd          m1, m7
    pmaddwd          m2, m8
    paddd            m0, m1
    paddd            m0, m2
    psrldq           m3, m5, 2                  ; y=-1,x=[-1,+5]
    psrldq           m1, m5, 4                  ; y=-1,x=[-0,+5]
    psrldq           m4, m5, 6                  ; y=-1,x=[+1,+5]
    psrldq           m2, m5, 8                  ; y=-1,x=[+2,+5]
    punpcklwd        m3, m1
    punpcklwd        m4, m2
    pmaddwd          m3, m9
    pmaddwd          m4, m10
    paddd            m3, m4
    paddd            m0, m3

    ; luma component & rounding
    movu             m1, [bufyq+xq*4]
    movu             m2, [bufyq+xq*4+82*2]
    phaddw           m1, m2
    pshufd           m2, m1, q3232
    paddw            m1, m2
%if cpuflag(sse4)
    pmulhrsw         m1, m15
%else
    pmulhrsw         m1, [pw_8192]
%endif
    punpcklwd        m1, [pw_1]
    pmaddwd          m1, m12
    paddd            m0, m1

    movu             m1, [bufq+xq*2-4]      ; y=0,x=[-2,+5]
    pshufd           m2, m1, q3321
    pxor             m3, m3
    pcmpgtw          m3, m2
    punpcklwd        m2, m3                 ; y=0,x=[0,3] in dword
.x_loop_ar2_inner:
    pmaddwd          m3, m1, m11
    paddd            m3, m0
    psrldq           m0, 4                  ; shift top to next pixel
    psrad            m3, [fg_dataq+FGData.ar_coeff_shift]
    ; we do not need to packssdw since we only care about one value
    paddd            m3, m2
    packssdw         m3, m3
    pminsw           m3, m13
    pmaxsw           m3, m14
    psrldq           m1, 2
    pslldq           m3, 2
    psrldq           m2, 4
%if cpuflag(sse4)
    pblendw          m1, m3, 00000010b
%else
    pand             m1, m15
    pandn            m4, m15, m3
    por              m1, m4
%endif
    ; overwrite previous pixel, should be ok
    movd  [bufq+xq*2-2], m1
    inc              xq
    jz .x_loop_ar2_end
    test             xq, 3
    jnz .x_loop_ar2_inner
    jmp .x_loop_ar2

.x_loop_ar2_end:
    add            bufq, 82*2
    add           bufyq, 82*4
    dec              hd
    jg .y_loop_ar2
    RET

.ar3:
    DEFINE_ARGS buf, bufy, fg_data, uv, bdmax, shift
%if WIN64
    mov              r6, rsp
    and             rsp, ~15
    sub             rsp, 96
    %define         tmp  rsp
%else
    %define         tmp  rsp+stack_offset-120
%endif
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    imul            uvd, 28
    SPLATW          m12, [base+round_vals-12+shiftq*2]
    pxor            m13, m13
    pcmpgtw         m13, m12
    punpcklwd       m12, m13
    sar          bdmaxd, 1
    SPLATW          m14, bdmaxd                 ; max_grain
    pcmpeqw         m15, m15
%if !cpuflag(sse4)
    pcmpeqw         m11, m11
    psrldq          m11, 14
    pslldq          m11, 4
    pxor            m11, m15
%endif
    pxor            m15, m14                    ; min_grain
%if cpuflag(sse4)
    SPLATW          m11, [base+hmul_bits+4]
%endif

    ; cf from y=-3,x=-3 until y=-3,x=-2
    movu             m0, [fg_dataq+FGData.ar_coeffs_uv+uvq+ 0]
    pxor             m1, m1
    pcmpgtb          m1, m0
    punpckhbw        m2, m0, m1
    punpcklbw        m0, m1
    pshufd           m6, m0, q0000
    pshufd           m7, m0, q1111
    pshufd           m8, m0, q2222
    pshufd           m9, m0, q3333
    pshufd          m10, m2, q0000
    pshufd          m13, m2, q1111
    mova     [tmp+16*0], m6
    mova     [tmp+16*1], m7
    mova     [tmp+16*2], m8
    mova     [tmp+16*3], m9
    mova     [tmp+16*4], m10
    mova     [tmp+16*5], m13
    pshufd           m6, m2, q2222
    pshufd           m7, m2, q3333

    ; cf from y=-1,x=-1 to y=0,x=-1 + luma component
    movu             m0, [fg_dataq+FGData.ar_coeffs_uv+uvq+16]
    pxor             m1, m1
    pcmpgtb          m1, m0
    punpckhbw        m2, m0, m1                 ; luma
    punpcklbw        m0, m1
    pshufd          m10, m0, q3232
    psrldq          m13, m0, 10
    ; y=0,x=[-3 to -1] + "1.0" for current pixel
    pinsrw          m13, [base+round_vals-10+shiftq*2], 3
    ; y=-1,x=[-1 to +2]
    pshufd           m8, m0, q0000
    pshufd           m9, m0, q1111
    ; y=-1,x=+3 + luma
    punpcklwd       m10, m2
    pshufd          m10, m10, q0000

    DEFINE_ARGS buf, bufy, fg_data, h, unused, x
    sub            bufq, 2*(82*38+44-(82*3+41))
    add           bufyq, 2*(79+82*3)
    mov              hd, 35
.y_loop_ar3:
    mov              xq, -38

.x_loop_ar3:
    ; first line
    movu             m0, [bufq+xq*2-82*6-6+ 0]      ; y=-3,x=[-3,+4]
    movd             m1, [bufq+xq*2-82*6-6+16]      ; y=-3,x=[+5,+6]
    palignr          m2, m1, m0, 2                  ; y=-3,x=[-2,+5]
    palignr          m1, m1, m0, 12                 ; y=-3,x=[+3,+6]
    punpckhwd        m3, m0, m2                     ; y=-3,x=[+1/+2,+2/+3,+3/+4,+4/+5]
    punpcklwd        m0, m2                         ; y=-3,x=[-3/-2,-2/-1,-1/+0,+0/+1]
    shufps           m2, m0, m3, q1032              ; y=-3,x=[-1/+0,+0/+1,+1/+2,+2/+3]

    pmaddwd          m0, [tmp+0*16]
    pmaddwd          m2, [tmp+1*16]
    pmaddwd          m3, [tmp+2*16]
    paddd            m0, m2
    paddd            m0, m3                         ; first 6 x of top y

    ; second line [m0/1 are busy]
    movu             m2, [bufq+xq*2-82*4-6+ 0]      ; y=-2,x=[-3,+4]
    movd             m3, [bufq+xq*2-82*4-6+16]      ; y=-2,x=[+5,+6]
    punpcklwd        m1, m2                         ; y=-3/-2,x=[+3/-3,+4/-2,+5/-1,+6/+0]
    palignr          m4, m3, m2, 2                  ; y=-2,x=[-2,+5]
    palignr          m3, m3, m2, 4                  ; y=-2,x=[-2,+5]
    punpckhwd        m5, m4, m3                     ; y=-2,x=[+2/+3,+3/+4,+4/+5,+5/+6]
    punpcklwd        m4, m3                         ; y=-2,x=[-2/-1,-1/+0,+0/+1,+1/+2]
    shufps           m3, m4, m5, q1032              ; t=-2,x=[+0/+1,+1/+2,+2/+3,+3/+4]
    pmaddwd          m1, [tmp+3*16]
    pmaddwd          m4, [tmp+4*16]
    pmaddwd          m3, [tmp+5*16]
    pmaddwd          m5, m6
    paddd            m1, m4
    paddd            m3, m5
    paddd            m0, m1
    paddd            m0, m3                         ; top 2 lines

    ; third line [m0 is busy] & luma + round
    movu             m1, [bufq+xq*2-82*2-6+ 0]      ; y=-1,x=[-3,+4]
    movd             m2, [bufq+xq*2-82*2-6+16]      ; y=-1,x=[+5,+6]
    movu             m5, [bufyq+xq*4]
    movu             m4, [bufyq+xq*4+82*2]
    phaddw           m5, m4
    palignr          m3, m2, m1, 2                  ; y=-1,x=[-2,+5]
    palignr          m2, m2, m1, 12                 ; y=-1,x=[+3,+6]
    pshufd           m4, m5, q3232
    paddw            m5, m4
%if cpuflag(sse4)
    pmulhrsw         m5, m11
%else
    pmulhrsw         m5, [pw_8192]
%endif
    punpckhwd        m4, m1, m3                     ; y=-1,x=[+1/+2,+2/+3,+3/+4,+4/+5]
    punpcklwd        m1, m3                         ; y=-1,x=[-3/-2,-2/-1,-1/+0,+0/+1]
    shufps           m3, m1, m4, q1032              ; y=-1,x=[-1/+0,+0/+1,+1/+2,+2/+3]
    punpcklwd        m2, m5
    pmaddwd          m1, m7
    pmaddwd          m3, m8
    pmaddwd          m4, m9
    pmaddwd          m2, m10
    paddd            m1, m3
    paddd            m4, m2
    paddd            m0, m12                        ; += round
    paddd            m1, m4
    paddd            m0, m1

    movu             m1, [bufq+xq*2-6]      ; y=0,x=[-3,+4]
.x_loop_ar3_inner:
    pmaddwd          m2, m1, m13
    pshufd           m3, m2, q1111
    paddd            m2, m3                 ; left+cur
    paddd            m2, m0                 ; add top
    psrldq           m0, 4
    psrad            m2, [fg_dataq+FGData.ar_coeff_shift]
    packssdw         m2, m2
    pminsw           m2, m14
    pmaxsw           m2, m15
    pslldq           m2, 4
    psrldq           m1, 2
%if cpuflag(sse4)
    pblendw          m1, m2, 00000100b
%else
    pand             m1, m11
    pandn            m3, m11, m2
    por              m1, m3
%endif
    ; overwrite previous pixels, should be ok
    movq  [bufq+xq*2-4], m1
    inc              xq
    jz .x_loop_ar3_end
    test             xq, 3
    jnz .x_loop_ar3_inner
    jmp .x_loop_ar3

.x_loop_ar3_end:
    add            bufq, 82*2
    add           bufyq, 82*4
    dec              hd
    jg .y_loop_ar3
%if WIN64
    mov             rsp, r6
%endif
    RET

INIT_XMM ssse3
cglobal fgy_32x32xn_16bpc, 6, 15, 16, dst, src, stride, fg_data, w, scaling, grain_lut
    mov             r7d, [fg_dataq+FGData.scaling_shift]
    lea              r8, [pb_mask]
%define base r8-pb_mask
    SPLATW          m11, [base+mul_bits+r7*2-14]
    mov             r6d, [fg_dataq+FGData.clip_to_restricted_range]
    mov             r9d, r9m        ; bdmax
    sar             r9d, 11         ; is_12bpc
    inc             r9d
    mov            r10d, r6d
    imul           r10d, r9d
    dec             r9d
    SPLATW          m13, [base+min+r10*2]
    lea             r9d, [r9d*3]
    lea             r9d, [r6d*2+r9d]
    SPLATW          m12, [base+max+r9*2]
    SPLATW          m10, r9m

    pcmpeqw          m9, m9
    psraw            m7, m10, 1             ; max_grain
    pxor             m9, m7                 ; min_grain
%if !cpuflag(sse4)
    pcmpeqw          m6, m6
    pslldq           m6, 4
%endif
    SPLATD          m14, [pd_16]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused1, \
                sby, see

    movifnidn      sbyd, sbym
    test           sbyd, sbyd
    setnz           r7b
    test            r7b, byte [fg_dataq+FGData.overlap_flag]
    jnz .vertical_overlap
    mov      dword sbym, 0

    imul           seed, sbyd, (173 << 24) | 37
    add            seed, (105 << 24) | 178
    rol            seed, 8
    movzx          seed, seew
    xor            seed, [fg_dataq+FGData.seed]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                unused1, unused2, see, src_bak

    lea        src_bakq, [srcq+wq*2]
    mov            r9mp, src_bakq
    neg              wq
    sub            dstq, srcq

.loop_x:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d                ; updated seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, src_bak

    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 164
    lea           offyq, [offyq+offxq*2+747] ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, src_bak

.loop_x_odd:
    movzx            hd, word hm
    mov      grain_lutq, grain_lutmp
.loop_y:
    ; src
    pand             m0, m10, [srcq+ 0]
    pand             m1, m10, [srcq+16]          ; m0-1: src as word

    ; scaling[src]
    vpgatherdw       m2, m0, scalingq-1, r11, r13, 8, 1, m4
    vpgatherdw       m3, m1, scalingq-1, r11, r13, 8, 1, m4
    REPX   {psrlw x, 8}, m2, m3

    ; grain = grain_lut[offy+y][offx+x]
    movu             m4, [grain_lutq+offxyq*2]
    movu             m5, [grain_lutq+offxyq*2+16]

    ; noise = round2(scaling[src] * grain, scaling_shift)
    REPX {pmullw x, m11}, m2, m3
    pmulhrsw         m4, m2
    pmulhrsw         m5, m3

    ; dst = clip_pixel(src, noise)
    paddw            m0, m4
    paddw            m1, m5
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova [dstq+srcq+ 0], m0
    mova [dstq+srcq+16], m1

    add            srcq, strideq
    add      grain_lutq, 82*2
    dec              hd
    jg .loop_y

    add              wq, 16
    jge .end
    mov        src_bakq, r9mp
    lea            srcq, [src_bakq+wq*2]
    btc        dword hm, 16
    jc .next_blk
    add          offxyd, 16
    cmp       dword r8m, 0
    je .loop_x_odd
    SPLATD          m15, [pw_27_17_17_27]
    add            r12d, 16                 ; top_offxy += 16
    jmp .loop_x_odd_v_overlap

.next_blk:
    cmp byte [fg_dataq+FGData.overlap_flag], 0
    je .loop_x

    ; r8m = sbym
    movq            m15, [pw_27_17_17_27]
    cmp       dword r8m, 0
    jne .loop_x_hv_overlap

    ; horizontal overlap (without vertical overlap)
.loop_x_h_overlap:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d                ; updated seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, src_bak, left_offxy

    lea     left_offxyd, [offyd+16]         ; previous column's offy*stride+offx
    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 164
    lea           offyq, [offyq+offxq*2+747] ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, src_bak, left_offxy

    movzx            hd, word hm
    mov      grain_lutq, grain_lutmp
.loop_y_h_overlap:
    ; grain = grain_lut[offy+y][offx+x]
    movu             m4, [grain_lutq+offxyq*2]
    movd             m5, [grain_lutq+left_offxyq*2]
    punpcklwd        m5, m4
    pmaddwd          m5, m15
    paddd            m5, m14
    psrad            m5, 5
    packssdw         m5, m5
%if cpuflag(sse4)
    pblendw          m4, m5, 00000011b
%else
    pand             m4, m6
    pandn            m0, m6, m5
    por              m4, m0
%endif
    pminsw           m4, m7
    pmaxsw           m4, m9

    ; src
    pand             m0, m10, [srcq+ 0]
    pand             m1, m10, [srcq+16]          ; m0-1: src as word

    ; scaling[src]
    vpgatherdw       m2, m0, scalingq-1, r13, r14, 8, 1, m5
    vpgatherdw       m3, m1, scalingq-1, r13, r14, 8, 1, m5
    REPX   {psrlw x, 8}, m2, m3

    ; noise = round2(scaling[src] * grain, scaling_shift)
    movu             m5, [grain_lutq+offxyq*2+16]
    REPX {pmullw x, m11}, m2, m3
    pmulhrsw         m4, m2
    pmulhrsw         m5, m3

    ; dst = clip_pixel(src, noise)
    paddw            m0, m4
    paddw            m1, m5
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova [dstq+srcq+ 0], m0
    mova [dstq+srcq+16], m1

    add            srcq, strideq
    add      grain_lutq, 82*2
    dec              hd
    jg .loop_y_h_overlap

    add              wq, 16
    jge .end
    mov        src_bakq, r9mp
    lea            srcq, [src_bakq+wq*2]
    or         dword hm, 0x10000
    add          offxyd, 16

    ; r8m = sbym
    cmp       dword r8m, 0
    je .loop_x_odd
    SPLATD          m15, [pw_27_17_17_27]
    add            r12d, 16                 ; top_offxy += 16
    jmp .loop_x_odd_v_overlap

.end:
    RET

.vertical_overlap:
    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused1, \
                sby, see

    movzx          sbyd, sbyb
    imul           seed, [fg_dataq+FGData.seed], 0x00010001
    imul            r7d, sbyd, 173 * 0x00010001
    imul           sbyd, 37 * 0x01000100
    add             r7d, (105 << 16) | 188
    add            sbyd, (178 << 24) | (141 << 8)
    and             r7d, 0x00ff00ff
    and            sbyd, 0xff00ff00
    xor            seed, r7d
    xor            seed, sbyd               ; (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                unused1, unused2, see, src_bak

    lea        src_bakq, [srcq+wq*2]
    mov            r9mp, src_bakq
    neg              wq
    sub            dstq, srcq

.loop_x_v_overlap:
    SPLATD          m15, [pw_27_17_17_27]

    ; we assume from the block above that bits 8-15 of r7d are zero'ed
    mov             r6d, seed
    or             seed, 0xeff4eff4
    test           seeb, seeh
    setp            r7b                     ; parity of top_seed
    shr            seed, 16
    shl             r7d, 16
    test           seeb, seeh
    setp            r7b                     ; parity of cur_seed
    or              r6d, 0x00010001
    xor             r7d, r6d
    mov            seed, r7d
    ror            seed, 1                  ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, src_bak, unused, top_offxy

    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    ror           offxd, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 164
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq*2+0x10001*747+32*82]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, src_bak, unused, top_offxy

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

.loop_x_odd_v_overlap:
    movzx            hd, word hm
    mov      grain_lutq, grain_lutmp
.loop_y_v_overlap:
    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq*2]
    movu             m2, [grain_lutq+top_offxyq*2]
    punpckhwd        m4, m2, m3
    punpcklwd        m2, m3
    REPX {pmaddwd x, m15}, m4, m2
    REPX {paddd   x, m14}, m4, m2
    REPX {psrad   x, 5}, m4, m2
    packssdw         m2, m4
    pminsw           m2, m7
    pmaxsw           m2, m9
    movu             m4, [grain_lutq+offxyq*2+16]
    movu             m3, [grain_lutq+top_offxyq*2+16]
    punpckhwd        m5, m3, m4
    punpcklwd        m3, m4
    REPX {pmaddwd x, m15}, m5, m3
    REPX {paddd   x, m14}, m5, m3
    REPX {psrad   x, 5}, m5, m3
    packssdw         m3, m5
    pminsw           m3, m7
    pmaxsw           m3, m9

    ; src
    pand             m0, m10, [srcq+ 0]          ; m0-1: src as word
    pand             m1, m10, [srcq+16]          ; m0-1: src as word

    ; scaling[src]
    ; noise = round2(scaling[src] * grain, scaling_shift)
    vpgatherdw       m4, m0, scalingq-1, r11, r13, 8, 1, m5
    psrlw            m4, 8
    pmullw           m4, m11
    pmulhrsw         m4, m2
    vpgatherdw       m5, m1, scalingq-1, r11, r13, 8, 1, m2
    psrlw            m5, 8
    pmullw           m5, m11
    pmulhrsw         m5, m3

    ; dst = clip_pixel(src, noise)
    paddw            m0, m4
    paddw            m1, m5
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova [dstq+srcq+ 0], m0
    mova [dstq+srcq+16], m1

    SPLATD          m15, [pw_27_17_17_27+4] ; swap weights for second v-overlap line
    add            srcq, strideq
    add      grain_lutq, 82*2
    dec              hw
    jz .end_y_v_overlap
    ; 2 lines get vertical overlap, then fall back to non-overlap code for
    ; remaining (up to) 30 lines
    xor              hd, 0x10000
    test             hd, 0x10000
    jnz .loop_y_v_overlap
    jmp .loop_y

.end_y_v_overlap:
    add              wq, 16
    jge .end_hv
    mov        src_bakq, r9mp
    lea            srcq, [src_bakq+wq*2]
    btc        dword hm, 16
    jc .next_blk_v
    SPLATD          m15, [pw_27_17_17_27]
    add          offxyd, 16
    add      top_offxyd, 16
    jmp .loop_x_odd_v_overlap

.next_blk_v:
    ; since fg_dataq.overlap is guaranteed to be set, we never jump
    ; back to .loop_x_v_overlap, and instead always fall-through to
    ; h+v overlap

    movq            m15, [pw_27_17_17_27]
.loop_x_hv_overlap:
    SPLATD           m8, [pw_27_17_17_27]

    ; we assume from the block above that bits 8-15 of r7d are zero'ed
    mov             r6d, seed
    or             seed, 0xeff4eff4
    test           seeb, seeh
    setp            r7b                     ; parity of top_seed
    shr            seed, 16
    shl             r7d, 16
    test           seeb, seeh
    setp            r7b                     ; parity of cur_seed
    or              r6d, 0x00010001
    xor             r7d, r6d
    mov            seed, r7d
    ror            seed, 1                  ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, src_bak, left_offxy, top_offxy, topleft_offxy

    lea  topleft_offxyq, [top_offxyq+16]
    lea     left_offxyq, [offyq+16]
    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    ror           offxd, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 164
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq*2+0x10001*747+32*82]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, src_bak, left_offxy, top_offxy, topleft_offxy

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

    movzx            hd, word hm
    mov      grain_lutq, grain_lutmp
.loop_y_hv_overlap:
    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq*2]
    movu             m5, [grain_lutq+top_offxyq*2]
    movd             m4, [grain_lutq+left_offxyq*2]
    movd             m2, [grain_lutq+topleft_offxyq*2]
    ; do h interpolation first (so top | top/left -> top, left | cur -> cur)
    punpcklwd        m4, m3
    punpcklwd        m2, m5
    REPX {pmaddwd x, m15}, m4, m2
    REPX {paddd   x, m14}, m4, m2
    REPX {psrad   x, 5}, m4, m2
    REPX {packssdw x, x}, m4, m2
    REPX {pminsw x, m7}, m4, m2
    REPX {pmaxsw x, m9}, m4, m2
%if cpuflag(sse4)
    pblendw          m3, m4, 00000011b
    pblendw          m5, m2, 00000011b
%else
    pand             m3, m6
    pand             m5, m6
    pandn            m0, m6, m4
    pandn            m1, m6, m2
    por              m3, m0
    por              m5, m1
%endif
    ; followed by v interpolation (top | cur -> cur)
    movu             m0, [grain_lutq+offxyq*2+16]
    movu             m1, [grain_lutq+top_offxyq*2+16]
    punpcklwd        m2, m5, m3
    punpckhwd        m5, m3
    punpcklwd        m3, m1, m0
    punpckhwd        m1, m0
    REPX {pmaddwd x, m8}, m2, m5, m3, m1
    REPX {paddd   x, m14}, m2, m5, m3, m1
    REPX {psrad   x, 5}, m2, m5, m3, m1
    packssdw         m2, m5
    packssdw         m3, m1
    REPX {pminsw x, m7}, m2, m3
    REPX {pmaxsw x, m9}, m2, m3

    ; src
    pand             m0, m10, [srcq+ 0]
    pand             m1, m10, [srcq+16]          ; m0-1: src as word

    ; scaling[src]
    ; noise = round2(scaling[src] * grain, scaling_shift)
    vpgatherdw       m4, m0, scalingq-1, r14, r10, 8, 1, m5
    psrlw            m4, 8
    pmullw           m4, m11
    pmulhrsw         m2, m4
    vpgatherdw       m5, m1, scalingq-1, r14, r10, 8, 1, m4
    psrlw            m5, 8
    pmullw           m5, m11
    pmulhrsw         m3, m5

    ; dst = clip_pixel(src, noise)
    paddw            m0, m2
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova [dstq+srcq+ 0], m0
    mova [dstq+srcq+16], m1

    SPLATD           m8, [pw_27_17_17_27+4] ; swap weights for second v-overlap line
    add            srcq, strideq
    add      grain_lutq, 82*2
    dec              hw
    jz .end_y_hv_overlap
    ; 2 lines get vertical overlap, then fall back to non-overlap code for
    ; remaining (up to) 30 lines
    xor              hd, 0x10000
    test             hd, 0x10000
    jnz .loop_y_hv_overlap
    jmp .loop_y_h_overlap

.end_y_hv_overlap:
    or         dword hm, 0x10000
    add              wq, 16
    jge .end_hv
    SPLATD          m15, [pw_27_17_17_27]
    add          offxyd, 16
    add      top_offxyd, 16
    mov        src_bakq, r9mp
    lea            srcq, [src_bakq+wq*2]
    jmp .loop_x_odd_v_overlap

.end_hv:
    RET

cglobal fguv_32x32xn_i420_16bpc, 6, 15, 16, dst, src, stride, fg_data, w, scaling, \
                                      grain_lut, h, sby, luma, lstride, uv_pl, is_id
%define base r8-pb_mask
    lea              r8, [pb_mask]
    mov             r7d, [fg_dataq+FGData.scaling_shift]
    SPLATW          m11, [base+mul_bits+r7*2-14]
    mov             r6d, [fg_dataq+FGData.clip_to_restricted_range]
    mov             r9d, r13m               ; bdmax
    sar             r9d, 11                 ; is_12bpc
    inc             r9d
    mov            r10d, r6d
    imul           r10d, r9d
    dec             r9d
    SPLATW          m13, [base+min+r10*2]
    lea            r10d, [r9d*3]
    mov            r11d, is_idm
    inc            r11d
    imul            r6d, r11d
    add            r10d, r6d
    SPLATW          m12, [base+max+r10*2]
    SPLATW          m10, r13m
%if cpuflag(sse4)
    pxor             m2, m2
%define mzero m2
%else
%define mzero m7
%endif
    mov           r13mp, strideq

    pcmpeqw          m8, m8
    psraw            m9, m10, 1
    pxor             m8, m9
%if !cpuflag(sse4)
    pcmpeqw          m2, m2
    pslldq           m2, 2
%endif

    cmp byte [fg_dataq+FGData.chroma_scaling_from_luma], 0
    jne .csfl

%macro FGUV_32x32xN_LOOP 1 ; not-csfl
    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused, sby, see, overlap

%if %1
    mov             r7d, r11m
    SPLATW           m0, [fg_dataq+FGData.uv_mult+r7*4]
    SPLATW           m1, [fg_dataq+FGData.uv_luma_mult+r7*4]
    punpcklwd       m14, m1, m0
    SPLATW          m15, [fg_dataq+FGData.uv_offset+r7*4]
    SPLATD           m7, [base+pw_4+r9*4]
    pmullw          m15, m7
%else
    SPLATD          m14, [pd_16]
    SPLATD          m15, [pw_23_22]
%endif

    movifnidn      sbyd, sbym
    test           sbyd, sbyd
    setnz           r7b
    test            r7b, byte [fg_dataq+FGData.overlap_flag]
    jnz %%vertical_overlap

    imul           seed, sbyd, (173 << 24) | 37
    add            seed, (105 << 24) | 178
    rol            seed, 8
    movzx          seed, seew
    xor            seed, [fg_dataq+FGData.seed]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                unused2, unused3, see, unused4, unused5, unused6, luma, lstride

    mov           lumaq, r9mp
    mov        lstrideq, r10mp
    lea             r10, [srcq+wq*2]
    lea             r11, [dstq+wq*2]
    lea             r12, [lumaq+wq*4]
    mov           r10mp, r10
    mov           r11mp, r11
    mov           r12mp, r12
    neg              wq

%%loop_x:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d               ; updated seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, unused1, unused2, unused3, luma, lstride

    mov           offxd, seed
    mov           offyd, seed
    ror           offyd, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 82
    lea           offyq, [offyq+offxq+498]  ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, unused1, unused2, unused3, luma, lstride

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
%%loop_y:
    ; src
    mova             m0, [srcq]
    mova             m1, [srcq+16]          ; m0-1: src as word

    ; luma_src
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    mova             m4, [lumaq+lstrideq*0+ 0]
    mova             m6, [lumaq+lstrideq*0+32]
    phaddw           m4, [lumaq+lstrideq*0+16]
    phaddw           m6, [lumaq+lstrideq*0+48]
    pavgw            m4, mzero
    pavgw            m6, mzero

%if %1
    punpckhwd        m3, m4, m0
    punpcklwd        m4, m0
    punpckhwd        m5, m6, m1
    punpcklwd        m6, m1                 ; { luma, chroma }
    REPX {pmaddwd x, m14}, m3, m4, m5, m6
    REPX {psrad   x, 6}, m3, m4, m5, m6
    packssdw         m4, m3
    packssdw         m6, m5
    REPX {paddw x, m15}, m4, m6
    REPX {pmaxsw x, mzero}, m4, m6
    REPX {pminsw x, m10}, m4, m6             ; clip_pixel()
%else
    REPX  {pand x, m10}, m4, m6
%endif

    ; scaling[luma_src]
    vpgatherdw       m3, m4, scalingq-1, r10, r12, 8, 1
    vpgatherdw       m5, m6, scalingq-1, r10, r12, 8, 1
    REPX   {psrlw x, 8}, m3, m5

    ; grain = grain_lut[offy+y][offx+x]
    movu             m4, [grain_lutq+offxyq*2]
    movu             m6, [grain_lutq+offxyq*2+16]

    ; noise = round2(scaling[luma_src] * grain, scaling_shift)
    REPX {pmullw x, m11}, m3, m5
    pmulhrsw         m4, m3
    pmulhrsw         m6, m5

    ; dst = clip_pixel(src, noise)
    paddw            m0, m4
    paddw            m1, m6
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova      [dstq+ 0], m0
    mova      [dstq+16], m1

    add            srcq, r13mp
    add            dstq, r13mp
    lea           lumaq, [lumaq+lstrideq*2]
    add      grain_lutq, 82*2
    dec              hb
    jg %%loop_y

    add              wq, 16
    jge %%end
    mov            srcq, r10mp
    mov            dstq, r11mp
    mov           lumaq, r12mp
    lea            srcq, [srcq+wq*2]
    lea            dstq, [dstq+wq*2]
    lea           lumaq, [lumaq+wq*4]
    cmp byte [fg_dataq+FGData.overlap_flag], 0
    je %%loop_x

    ; r8m = sbym
    cmp       dword r8m, 0
    jne %%loop_x_hv_overlap

    ; horizontal overlap (without vertical overlap)
%%loop_x_h_overlap:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d               ; updated seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, left_offxy, unused1, unused2, luma, lstride

    lea     left_offxyd, [offyd+16]         ; previous column's offy*stride+offx
    mov           offxd, seed
    mov           offyd, seed
    ror           offyd, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 82
    lea           offyq, [offyq+offxq+498]  ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, left_offxy, unused1, unused2, luma, lstride

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
%%loop_y_h_overlap:
    mova             m0, [srcq]
    mova             m1, [srcq+16]

    ; luma_src
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    mova             m4, [lumaq+lstrideq*0+ 0]
    mova             m6, [lumaq+lstrideq*0+32]
    phaddw           m4, [lumaq+lstrideq*0+16]
    phaddw           m6, [lumaq+lstrideq*0+48]
    pavgw            m4, mzero
    pavgw            m6, mzero

%if %1
    punpckhwd        m3, m4, m0
    punpcklwd        m4, m0
    punpckhwd        m5, m6, m1
    punpcklwd        m6, m1                 ; { luma, chroma }
    REPX {pmaddwd x, m14}, m3, m4, m5, m6
    REPX {psrad   x, 6}, m3, m4, m5, m6
    packssdw         m4, m3
    packssdw         m6, m5
    REPX {paddw x, m15}, m4, m6
    REPX {pmaxsw x, mzero}, m4, m6
    REPX {pminsw x, m10}, m4, m6             ; clip_pixel()
%else
    REPX  {pand x, m10}, m4, m6
%endif

    ; grain = grain_lut[offy+y][offx+x]
    movu             m7, [grain_lutq+offxyq*2]
    movd             m5, [grain_lutq+left_offxyq*2+ 0]
    punpcklwd        m5, m7                ; {left0, cur0}
%if %1
    pmaddwd          m5, [pw_23_22]
    paddd            m5, [pd_16]
%else
    pmaddwd          m5, m15
    paddd            m5, m14
%endif
    psrad            m5, 5
    packssdw         m5, m5
    pmaxsw           m5, m8
    pminsw           m5, m9
%if cpuflag(sse4)
    pblendw          m5, m7, 11111110b
%else
    pand             m7, m2
    pandn            m3, m2, m5
    por              m5, m7, m3
%endif
    movu             m3, [grain_lutq+offxyq*2+16]

    ; scaling[luma_src]
    vpgatherdw       m7, m4, scalingq-1, r2, r12, 8, 1
    vpgatherdw       m4, m6, scalingq-1, r2, r12, 8, 1
    REPX   {psrlw x, 8}, m7, m4

    ; noise = round2(scaling[luma_src] * grain, scaling_shift)
    REPX {pmullw x, m11}, m7, m4
    pmulhrsw         m5, m7
    pmulhrsw         m3, m4

    ; dst = clip_pixel(src, noise)
    paddw            m0, m5
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova      [dstq+ 0], m0
    mova      [dstq+16], m1

    add            srcq, r13mp
    add            dstq, r13mp
    lea           lumaq, [lumaq+lstrideq*2]
    add      grain_lutq, 82*2
    dec              hb
    jg %%loop_y_h_overlap

    add              wq, 16
    jge %%end
    mov            srcq, r10mp
    mov            dstq, r11mp
    mov           lumaq, r12mp
    lea            srcq, [srcq+wq*2]
    lea            dstq, [dstq+wq*2]
    lea           lumaq, [lumaq+wq*4]

    ; r8m = sbym
    cmp       dword r8m, 0
    jne %%loop_x_hv_overlap
    jmp %%loop_x_h_overlap

%%end:
    RET

%%vertical_overlap:
    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused, \
                sby, see, unused1, unused2, unused3, lstride

    movzx          sbyd, sbyb
    imul           seed, [fg_dataq+FGData.seed], 0x00010001
    imul            r7d, sbyd, 173 * 0x00010001
    imul           sbyd, 37 * 0x01000100
    add             r7d, (105 << 16) | 188
    add            sbyd, (178 << 24) | (141 << 8)
    and             r7d, 0x00ff00ff
    and            sbyd, 0xff00ff00
    xor            seed, r7d
    xor            seed, sbyd               ; (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                unused1, unused2, see, unused3, unused4, unused5, luma, lstride

    mov           lumaq, r9mp
    mov        lstrideq, r10mp
    lea             r10, [srcq+wq*2]
    lea             r11, [dstq+wq*2]
    lea             r12, [lumaq+wq*4]
    mov           r10mp, r10
    mov           r11mp, r11
    mov           r12mp, r12
    neg              wq

%%loop_x_v_overlap:
    ; we assume from the block above that bits 8-15 of r7d are zero'ed
    mov             r6d, seed
    or             seed, 0xeff4eff4
    test           seeb, seeh
    setp            r7b                     ; parity of top_seed
    shr            seed, 16
    shl             r7d, 16
    test           seeb, seeh
    setp            r7b                     ; parity of cur_seed
    or              r6d, 0x00010001
    xor             r7d, r6d
    mov            seed, r7d
    ror            seed, 1                  ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, unused1, top_offxy, unused2, luma, lstride

    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    ror           offxd, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 82
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq+0x10001*498+16*82]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, unused1, top_offxy, unused2, luma, lstride

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
%%loop_y_v_overlap:
    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq*2]
    movu             m5, [grain_lutq+top_offxyq*2]
    punpckhwd        m7, m5, m3
    punpcklwd        m5, m3                 ; {top/cur interleaved}
%if %1
    REPX {pmaddwd x, [pw_23_22]}, m7, m5
    REPX  {paddd x, [pd_16]}, m7, m5
%else
    REPX {pmaddwd x, m15}, m7, m5
    REPX  {paddd x, m14}, m7, m5
%endif
    REPX   {psrad x, 5}, m7, m5
    packssdw         m3, m5, m7
    pmaxsw           m3, m8
    pminsw           m3, m9

    ; grain = grain_lut[offy+y][offx+x]
    movu             m4, [grain_lutq+offxyq*2+16]
    movu             m5, [grain_lutq+top_offxyq*2+16]
    punpckhwd        m7, m5, m4
    punpcklwd        m5, m4                 ; {top/cur interleaved}
%if %1
    REPX {pmaddwd x, [pw_23_22]}, m7, m5
    REPX  {paddd x, [pd_16]}, m7, m5
%else
    REPX {pmaddwd x, m15}, m7, m5
    REPX  {paddd x, m14}, m7, m5
%endif
    REPX   {psrad x, 5}, m7, m5
    packssdw         m4, m5, m7
    pmaxsw           m4, m8
    pminsw           m4, m9

    ; src
    mova             m0, [srcq]
    mova             m1, [srcq+16]

    ; luma_src
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    mova             m5, [lumaq+lstrideq*0+ 0]
    mova             m6, [lumaq+lstrideq*0+32]
    phaddw           m5, [lumaq+lstrideq*0+16]
    phaddw           m6, [lumaq+lstrideq*0+48]
    pavgw            m5, mzero
    pavgw            m6, mzero

%if %1
    punpckhwd        m7, m5, m0
    punpcklwd        m5, m0
    REPX {pmaddwd x, m14}, m7, m5
    REPX {psrad   x, 6}, m7, m5
    packssdw         m5, m7
    punpckhwd        m7, m6, m1
    punpcklwd        m6, m1                 ; { luma, chroma }
    REPX {pmaddwd x, m14}, m7, m6
    REPX {psrad   x, 6}, m7, m6
    packssdw         m6, m7
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    REPX {paddw x, m15}, m5, m6
    REPX {pmaxsw x, mzero}, m5, m6
    REPX {pminsw x, m10}, m5, m6            ; clip_pixel()
%else
    REPX  {pand x, m10}, m5, m6
%endif

    ; scaling[luma_src]
    vpgatherdw       m7, m5, scalingq-1, r10, r12, 8, 1
    vpgatherdw       m5, m6, scalingq-1, r10, r12, 8, 1
    REPX   {psrlw x, 8}, m7, m5

    ; noise = round2(scaling[luma_src] * grain, scaling_shift)
    REPX {pmullw x, m11}, m7, m5
    pmulhrsw         m3, m7
    pmulhrsw         m4, m5

    ; dst = clip_pixel(src, noise)
    paddw            m0, m3
    paddw            m1, m4
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova      [dstq+ 0], m0
    mova      [dstq+16], m1

    dec              hb
    jle %%end_y_v_overlap
    add            srcq, r13mp
    add            dstq, r13mp
    lea           lumaq, [lumaq+lstrideq*2]
    add      grain_lutq, 82*2
    jmp %%loop_y

%%end_y_v_overlap:
    add              wq, 16
    jge %%end_hv
    mov            srcq, r10mp
    mov            dstq, r11mp
    mov           lumaq, r12mp
    lea            srcq, [srcq+wq*2]
    lea            dstq, [dstq+wq*2]
    lea           lumaq, [lumaq+wq*4]

    ; since fg_dataq.overlap is guaranteed to be set, we never jump
    ; back to .loop_x_v_overlap, and instead always fall-through to
    ; h+v overlap

%%loop_x_hv_overlap:
    ; we assume from the block above that bits 8-15 of r7d are zero'ed
    mov             r6d, seed
    or             seed, 0xeff4eff4
    test           seeb, seeh
    setp            r7b                     ; parity of top_seed
    shr            seed, 16
    shl             r7d, 16
    test           seeb, seeh
    setp            r7b                     ; parity of cur_seed
    or              r6d, 0x00010001
    xor             r7d, r6d
    mov            seed, r7d
    ror            seed, 1                  ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                offx, offy, see, left_offxy, top_offxy, topleft_offxy, luma, lstride

    lea  topleft_offxyq, [top_offxyq+16]
    lea     left_offxyq, [offyq+16]
    mov           offyd, seed
    mov           offxd, seed
    ror           offyd, 8
    ror           offxd, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 82
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq+0x10001*498+16*82]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, \
                h, offxy, see, left_offxy, top_offxy, topleft_offxy, luma, lstride

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
%%loop_y_hv_overlap:
    ; grain = grain_lut[offy+y][offx+x]
    movd             m5, [grain_lutq+left_offxyq*2]
    pinsrw           m5, [grain_lutq+topleft_offxyq*2], 1   ; { left, top/left }
    movu             m3, [grain_lutq+offxyq*2]
    movu             m4, [grain_lutq+top_offxyq*2]
    punpcklwd        m7, m3, m4             ; { cur0, top0 }
    punpcklwd        m5, m7                 ; { cur/left } interleaved
%if %1
    pmaddwd          m5, [pw_23_22]
    paddd            m5, [pd_16]
%else
    pmaddwd          m5, m15
    paddd            m5, m14
%endif
    psrad            m5, 5
    packssdw         m5, m5
    pmaxsw           m5, m8
    pminsw           m5, m9
%if cpuflag(sse4)
    pblendw          m3, m5, 00000001b
    psrldq           m5, 2
    pblendw          m5, m4, 11111110b
%else
    pand             m3, m2
    pandn            m7, m2, m5
    por              m3, m7
    psrldq           m5, 2
    pand             m4, m2
    pandn            m7, m2, m5
    por              m5, m4, m7
%endif

    punpckhwd        m7, m5, m3
    punpcklwd        m5, m3                 ; {top/cur interleaved}
%if %1
    REPX {pmaddwd x, [pw_23_22]}, m7, m5
    REPX  {paddd x, [pd_16]}, m5, m7
%else
    REPX {pmaddwd x, m15}, m7, m5
    REPX  {paddd x, m14}, m5, m7
%endif
    REPX   {psrad x, 5}, m5, m7
    packssdw         m3, m5, m7
    pmaxsw           m3, m8
    pminsw           m3, m9

    ; right half
    movu             m4, [grain_lutq+offxyq*2+16]
    movu             m0, [grain_lutq+top_offxyq*2+16]
    punpckhwd        m1, m0, m4
    punpcklwd        m0, m4                 ; {top/cur interleaved}
%if %1
    REPX {pmaddwd x, [pw_23_22]}, m1, m0
    REPX  {paddd x, [pd_16]}, m1, m0
%else
    REPX {pmaddwd x, m15}, m1, m0
    REPX  {paddd x, m14}, m1, m0
%endif
    REPX   {psrad x, 5}, m1, m0
    packssdw         m4, m0, m1
    pmaxsw           m4, m8
    pminsw           m4, m9

    ; src
    mova             m0, [srcq]
    mova             m1, [srcq+16]

    ; luma_src
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    mova             m6, [lumaq+lstrideq*0+ 0]
    mova             m5, [lumaq+lstrideq*0+32]
    phaddw           m6, [lumaq+lstrideq*0+16]
    phaddw           m5, [lumaq+lstrideq*0+48]
    pavgw            m6, mzero
    pavgw            m5, mzero

%if %1
    punpckhwd        m7, m6, m0
    punpcklwd        m6, m0
    REPX {pmaddwd x, m14}, m7, m6
    REPX {psrad   x, 6}, m7, m6
    packssdw         m6, m7
    punpckhwd        m7, m5, m1
    punpcklwd        m5, m1                 ; { luma, chroma }
    REPX {pmaddwd x, m14}, m7, m5
    REPX {psrad   x, 6}, m7, m5
    packssdw         m5, m7
%if !cpuflag(sse4)
    pxor          mzero, mzero
%endif
    REPX {paddw x, m15}, m6, m5
    REPX {pmaxsw x, mzero}, m6, m5
    REPX {pminsw x, m10}, m6, m5            ; clip_pixel()
%else
    REPX  {pand x, m10}, m6, m5
%endif

    ; scaling[luma_src]
    vpgatherdw       m7, m6, scalingq-1, r2, r12, 8, 1
    vpgatherdw       m6, m5, scalingq-1, r2, r12, 8, 1
    REPX   {psrlw x, 8}, m7, m6

    ; noise = round2(scaling[luma_src] * grain, scaling_shift)
    REPX {pmullw x, m11}, m7, m6
    pmulhrsw         m3, m7
    pmulhrsw         m4, m6

    ; dst = clip_pixel(src, noise)
    paddw            m0, m3
    paddw            m1, m4
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    mova      [dstq+ 0], m0
    mova      [dstq+16], m1

    add            srcq, r13mp
    add            dstq, r13mp
    lea           lumaq, [lumaq+lstrideq*2]
    add      grain_lutq, 82*2
    dec              hb
    jg %%loop_y_h_overlap

%%end_y_hv_overlap:
    add              wq, 16
    jge %%end_hv
    mov            srcq, r10mp
    mov            dstq, r11mp
    mov           lumaq, r12mp
    lea            srcq, [srcq+wq*2]
    lea            dstq, [dstq+wq*2]
    lea           lumaq, [lumaq+wq*4]
    jmp %%loop_x_hv_overlap

%%end_hv:
    RET
%endmacro

    FGUV_32x32xN_LOOP 1
.csfl:
    FGUV_32x32xN_LOOP 0

%endif ; ARCH_X86_64
