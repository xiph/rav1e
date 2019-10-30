; Copyright © 2019, VideoLAN and dav1d authors
; Copyright © 2019, Two Orioles, LLC
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

SECTION_RODATA
pb_mask: db 0, 0x80, 0x80, 0, 0x80, 0, 0, 0x80, 0x80, 0, 0, 0x80, 0, 0x80, 0x80, 0
rnd_next_upperbit_mask: dw 0x100B, 0x2016, 0x402C, 0x8058
byte_blend: db 0, 0, 0, 0xff, 0, 0, 0, 0
pw_1024: times 2 dw 1024
pd_m65536: dd ~0xffff
hmul_bits: dw 32768, 16384, 8192, 4096
round: dw 2048, 1024, 512
mul_bits: dw 256, 128, 64, 32, 16
round_vals: dw 32, 64, 128, 256, 512
max: dw 255, 235
min: dw 0, 16
pb_27_17_17_27: db 27, 17, 17, 27
pb_1: db 1

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

JMP_TABLE generate_grain_y_avx2, 0, 1, 2, 3

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
    .ar_coeffs_uv:              resb 2 * 26 ; includes padding
    .ar_coeff_shift:            resd 1
    .grain_scale_shift:         resd 1
    .uv_mult:                   resd 2
    .uv_luma_mult:              resd 2
    .uv_offset:                 resd 2
    .overlap_flag:              resd 1
    .clip_to_restricted_range:  resd 1
endstruc

cextern gaussian_sequence

SECTION .text

INIT_XMM avx2
cglobal generate_grain_y, 2, 9, 16, buf, fg_data
    lea              r4, [pb_mask]
%define base r4-pb_mask
    movq            xm1, [base+rnd_next_upperbit_mask]
    movq            xm4, [base+mul_bits]
    movq            xm7, [base+hmul_bits]
    mov             r2d, [fg_dataq+FGData.grain_scale_shift]
    vpbroadcastw    xm8, [base+round+r2*2]
    mova            xm5, [base+pb_mask]
    vpbroadcastw    xm0, [fg_dataq+FGData.seed]
    vpbroadcastd    xm9, [base+pd_m65536]
    mov              r2, -73*82
    sub            bufq, r2
    lea              r3, [gaussian_sequence]
.loop:
    pand            xm2, xm0, xm1
    psrlw           xm3, xm2, 10
    por             xm2, xm3            ; bits 0xf, 0x1e, 0x3c and 0x78 are set
    pmullw          xm2, xm4            ; bits 0x0f00 are set
    pshufb          xm2, xm5, xm2       ; set 15th bit for next 4 seeds
    psllq           xm6, xm2, 30
    por             xm2, xm6
    psllq           xm6, xm2, 15
    por             xm2, xm6            ; aggregate each bit into next seed's high bit
    pmulhuw         xm3, xm0, xm7
    por             xm2, xm3            ; 4 next output seeds
    pshuflw         xm0, xm2, q3333
    psrlw           xm2, 5
    pmovzxwd        xm3, xm2
    mova            xm6, xm9
    vpgatherdd      xm2, [r3+xm3*2], xm6
    pandn           xm2, xm9, xm2
    packusdw        xm2, xm2
    pmulhrsw        xm2, xm8
    packsswb        xm2, xm2
    movd      [bufq+r2], xm2
    add              r2, 4
    jl .loop

    ; auto-regression code
    movsxd           r2, [fg_dataq+FGData.ar_coeff_lag]
    movsxd           r2, [base+generate_grain_y_avx2_table+r2*4]
    lea              r2, [r2+base+generate_grain_y_avx2_table]
    jmp              r2

.ar1:
    DEFINE_ARGS buf, fg_data, cf3, shift, val3, min, max, x, val0
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    movsx          cf3d, byte [fg_dataq+FGData.ar_coeffs_y+3]
    movd            xm4, [fg_dataq+FGData.ar_coeffs_y]
    DEFINE_ARGS buf, h, cf3, shift, val3, min, max, x, val0
    pinsrb          xm4, [pb_1], 3
    pmovsxbw        xm4, xm4
    pshufd          xm5, xm4, q1111
    pshufd          xm4, xm4, q0000
    vpbroadcastw    xm3, [base+round_vals+shiftq*2-12]    ; rnd
    sub            bufq, 82*73-(82*3+79)
    mov              hd, 70
    mov            mind, -128
    mov            maxd, 127
.y_loop_ar1:
    mov              xq, -76
    movsx         val3d, byte [bufq+xq-1]
.x_loop_ar1:
    pmovsxbw        xm0, [bufq+xq-82-1]     ; top/left
    pmovsxbw        xm2, [bufq+xq-82+0]     ; top
    pmovsxbw        xm1, [bufq+xq-82+1]     ; top/right
    punpcklwd       xm0, xm2
    punpcklwd       xm1, xm3
    pmaddwd         xm0, xm4
    pmaddwd         xm1, xm5
    paddd           xm0, xm1
.x_loop_ar1_inner:
    movd          val0d, xm0
    psrldq          xm0, 4
    imul          val3d, cf3d
    add           val3d, val0d
%if WIN64
    sarx          val3d, val3d, shiftd
%else
    sar           val3d, shiftb
%endif
    movsx         val0d, byte [bufq+xq]
    add           val3d, val0d
    cmp           val3d, maxd
    cmovg         val3d, maxd
    cmp           val3d, mind
    cmovl         val3d, mind
    mov  byte [bufq+xq], val3b
    ; keep val3d in-place as left for next x iteration
    inc              xq
    jz .x_loop_ar1_end
    test             xq, 3
    jnz .x_loop_ar1_inner
    jmp .x_loop_ar1

.x_loop_ar1_end:
    add            bufq, 82
    dec              hd
    jg .y_loop_ar1
.ar0:
    RET

.ar2:
    DEFINE_ARGS buf, fg_data, shift
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    movd           xm14, [base+hmul_bits-10+shiftq*2]
    movq           xm15, [base+byte_blend+1]
    pmovsxbw        xm8, [fg_dataq+FGData.ar_coeffs_y+0]    ; cf0-7
    movd            xm9, [fg_dataq+FGData.ar_coeffs_y+8]    ; cf8-11
    pmovsxbw        xm9, xm9
    DEFINE_ARGS buf, h, x
    pshufd         xm12, xm9, q0000
    pshufd         xm13, xm9, q1111
    pshufd         xm11, xm8, q3333
    pshufd         xm10, xm8, q2222
    pshufd          xm9, xm8, q1111
    pshufd          xm8, xm8, q0000
    sub            bufq, 82*73-(82*3+79)
    mov              hd, 70
.y_loop_ar2:
    mov              xq, -76

.x_loop_ar2:
    pmovsxbw        xm0, [bufq+xq-82*2-2]   ; y=-2,x=[-2,+5]
    pmovsxbw        xm1, [bufq+xq-82*1-2]   ; y=-1,x=[-2,+5]
    psrldq          xm2, xm0, 2             ; y=-2,x=[-1,+5]
    psrldq          xm3, xm1, 2             ; y=-1,x=[-1,+5]
    psrldq          xm4, xm1, 4             ; y=-1,x=[+0,+5]
    punpcklwd       xm2, xm0, xm2
    punpcklwd       xm3, xm4
    pmaddwd         xm2, xm8
    pmaddwd         xm3, xm11
    paddd           xm2, xm3

    psrldq          xm4, xm0, 4             ; y=-2,x=[+0,+5]
    psrldq          xm5, xm0, 6             ; y=-2,x=[+1,+5]
    psrldq          xm6, xm0, 8             ; y=-2,x=[+2,+5]
    punpcklwd       xm4, xm5
    punpcklwd       xm6, xm1
    psrldq          xm7, xm1, 6             ; y=-1,x=[+1,+5]
    psrldq          xm1, xm1, 8             ; y=-1,x=[+2,+5]
    punpcklwd       xm7, xm1
    pmaddwd         xm4, xm9
    pmaddwd         xm6, xm10
    pmaddwd         xm7, xm12
    paddd           xm4, xm6
    paddd           xm2, xm7
    paddd           xm2, xm4

    movq            xm0, [bufq+xq-2]        ; y=0,x=[-2,+5]
.x_loop_ar2_inner:
    pmovsxbw        xm1, xm0
    pmaddwd         xm3, xm1, xm13
    paddd           xm3, xm2
    psrldq          xm1, 4                  ; y=0,x=0
    psrldq          xm2, 4                  ; shift top to next pixel
    psrad           xm3, 5
    packssdw        xm3, xm3
    pmulhrsw        xm3, xm14
    paddw           xm3, xm1
    packsswb        xm3, xm3
    pextrb    [bufq+xq], xm3, 0
    pslldq          xm3, 2
    pand            xm3, xm15
    pandn           xm0, xm15, xm0
    por             xm0, xm3
    psrldq          xm0, 1
    inc              xq
    jz .x_loop_ar2_end
    test             xq, 3
    jnz .x_loop_ar2_inner
    jmp .x_loop_ar2

.x_loop_ar2_end:
    add            bufq, 82
    dec              hd
    jg .y_loop_ar2
    RET

.ar3:
    DEFINE_ARGS buf, fg_data, shift
%if WIN64
    SUB             rsp, 16*12
%assign stack_size_padded (stack_size_padded+16*12)
%assign stack_size (stack_size+16*12)
%else
    ALLOC_STACK   16*12
%endif
    mov          shiftd, [fg_dataq+FGData.ar_coeff_shift]
    movd           xm14, [base+hmul_bits-10+shiftq*2]
    movq           xm15, [base+byte_blend]
    pmovsxbw        xm0, [fg_dataq+FGData.ar_coeffs_y+ 0]   ; cf0-7
    pmovsxbw        xm1, [fg_dataq+FGData.ar_coeffs_y+ 8]   ; cf8-15
    pmovsxbw        xm2, [fg_dataq+FGData.ar_coeffs_y+16]   ; cf16-23
    pshufd          xm9, xm0, q1111
    pshufd         xm10, xm0, q2222
    pshufd         xm11, xm0, q3333
    pshufd          xm0, xm0, q0000
    pshufd          xm6, xm1, q1111
    pshufd          xm7, xm1, q2222
    pshufd          xm8, xm1, q3333
    pshufd          xm1, xm1, q0000
    pshufd          xm3, xm2, q1111
    pshufd          xm4, xm2, q2222
    psrldq          xm5, xm2, 10
    pshufd          xm2, xm2, q0000
    pinsrw          xm5, [base+round_vals+shiftq*2-10], 3
    mova    [rsp+ 0*16], xm0
    mova    [rsp+ 1*16], xm9
    mova    [rsp+ 2*16], xm10
    mova    [rsp+ 3*16], xm11
    mova    [rsp+ 4*16], xm1
    mova    [rsp+ 5*16], xm6
    mova    [rsp+ 6*16], xm7
    mova    [rsp+ 7*16], xm8
    mova    [rsp+ 8*16], xm2
    mova    [rsp+ 9*16], xm3
    mova    [rsp+10*16], xm4
    mova    [rsp+11*16], xm5
    pxor           xm13, xm13
    DEFINE_ARGS buf, h, x
    sub            bufq, 82*73-(82*3+79)
    mov              hd, 70
.y_loop_ar3:
    mov              xq, -76

.x_loop_ar3:
    movu            xm0, [bufq+xq-82*3-3]   ; y=-3,x=[-3,+12]
    movu            xm1, [bufq+xq-82*2-3]   ; y=-2,x=[-3,+12]
    movu            xm2, [bufq+xq-82*1-3]   ; y=-1,x=[-3,+12]
    pxor            xm3, xm3
    pcmpgtb         xm6, xm3, xm2
    pcmpgtb         xm5, xm3, xm1
    pcmpgtb         xm4, xm3, xm0
    punpckhbw       xm3, xm0, xm4
    punpcklbw       xm0, xm4
    punpckhbw       xm4, xm1, xm5
    punpcklbw       xm1, xm5
    punpckhbw       xm5, xm2, xm6
    punpcklbw       xm2, xm6

    psrldq          xm6, xm0, 2
    psrldq          xm7, xm0, 4
    psrldq          xm8, xm0, 6
    psrldq          xm9, xm0, 8
    palignr        xm10, xm3, xm0, 10
    palignr        xm11, xm3, xm0, 12

    punpcklwd       xm0, xm6
    punpcklwd       xm7, xm8
    punpcklwd       xm9, xm10
    punpcklwd      xm11, xm1
    pmaddwd         xm0, [rsp+ 0*16]
    pmaddwd         xm7, [rsp+ 1*16]
    pmaddwd         xm9, [rsp+ 2*16]
    pmaddwd        xm11, [rsp+ 3*16]
    paddd           xm0, xm7
    paddd           xm9, xm11
    paddd           xm0, xm9

    psrldq          xm6, xm1, 2
    psrldq          xm7, xm1, 4
    psrldq          xm8, xm1, 6
    psrldq          xm9, xm1, 8
    palignr        xm10, xm4, xm1, 10
    palignr        xm11, xm4, xm1, 12
    psrldq         xm12, xm2, 2

    punpcklwd       xm6, xm7
    punpcklwd       xm8, xm9
    punpcklwd      xm10, xm11
    punpcklwd      xm12, xm2, xm12
    pmaddwd         xm6, [rsp+ 4*16]
    pmaddwd         xm8, [rsp+ 5*16]
    pmaddwd        xm10, [rsp+ 6*16]
    pmaddwd        xm12, [rsp+ 7*16]
    paddd           xm6, xm8
    paddd          xm10, xm12
    paddd           xm6, xm10
    paddd           xm0, xm6

    psrldq          xm6, xm2, 4
    psrldq          xm7, xm2, 6
    psrldq          xm8, xm2, 8
    palignr         xm9, xm5, xm2, 10
    palignr         xm5, xm5, xm2, 12

    punpcklwd       xm6, xm7
    punpcklwd       xm8, xm9
    punpcklwd       xm5, xm13
    pmaddwd         xm6, [rsp+ 8*16]
    pmaddwd         xm8, [rsp+ 9*16]
    pmaddwd         xm5, [rsp+10*16]
    paddd           xm0, xm6
    paddd           xm8, xm5
    paddd           xm0, xm8

    movq            xm1, [bufq+xq-3]        ; y=0,x=[-3,+4]
.x_loop_ar3_inner:
    pmovsxbw        xm2, xm1
    pmaddwd         xm2, [rsp+16*11]
    pshufd          xm3, xm2, q1111
    paddd           xm2, xm3                ; left+cur
    paddd           xm2, xm0                ; add top
    psrldq          xm0, 4
    psrad           xm2, 5
    packssdw        xm2, xm2
    pmulhrsw        xm2, xm14
    packsswb        xm2, xm2
    pextrb    [bufq+xq], xm2, 0
    pslldq          xm2, 3
    pand            xm2, xm15
    pandn           xm1, xm15, xm1
    por             xm1, xm2
    psrldq          xm1, 1
    inc              xq
    jz .x_loop_ar3_end
    test             xq, 3
    jnz .x_loop_ar3_inner
    jmp .x_loop_ar3

.x_loop_ar3_end:
    add            bufq, 82
    dec              hd
    jg .y_loop_ar3
    RET

INIT_YMM avx2
cglobal fgy_32x32xn, 6, 13, 16, dst, src, stride, fg_data, w, scaling, grain_lut
    pcmpeqw         m10, m10
    psrld           m10, 24
    mov             r7d, [fg_dataq+FGData.scaling_shift]
    lea              r8, [pb_mask]
%define base r8-pb_mask
    vpbroadcastw    m11, [base+mul_bits+r7*2-14]
    mov             r7d, [fg_dataq+FGData.clip_to_restricted_range]
    vpbroadcastw    m12, [base+max+r7*2]
    vpbroadcastw    m13, [base+min+r7*2]

    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused, sby, see, overlap

    mov        overlapd, [fg_dataq+FGData.overlap_flag]
    movifnidn      sbyd, sbym
    test           sbyd, sbyd
    setnz           r7b
    test            r7b, overlapb
    jnz .vertical_overlap

    imul           seed, sbyd, (173 << 24) | 37
    add            seed, (105 << 24) | 178
    rol            seed, 8
    movzx          seed, seew
    xor            seed, [fg_dataq+FGData.seed]


    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                unused1, unused2, see, overlap

    lea        src_bakq, [srcq+wq]
    neg              wq
    sub            dstq, srcq

.loop_x:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d                ; updated seed

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                offx, offy, see, overlap

    mov           offxd, seed
    rorx          offyd, seed, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 164
    lea           offyq, [offyq+offxq*2+747] ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                h, offxy, see, overlap

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
.loop_y:
    ; src
    mova             m0, [srcq]
    pxor             m2, m2
    punpckhbw        m1, m0, m2
    punpcklbw        m0, m2                 ; m0-1: src as word
    punpckhwd        m5, m0, m2
    punpcklwd        m4, m0, m2
    punpckhwd        m7, m1, m2
    punpcklwd        m6, m1, m2             ; m4-7: src as dword

    ; scaling[src]
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m8, [scalingq+m4], m3
    vpgatherdd       m4, [scalingq+m5], m9
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m5, [scalingq+m6], m3
    vpgatherdd       m6, [scalingq+m7], m9
    pand             m8, m10
    pand             m4, m10
    pand             m5, m10
    pand             m6, m10
    packusdw         m8, m4
    packusdw         m5, m6

    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq]
    pcmpgtb          m7, m2, m3
    punpcklbw        m2, m3, m7
    punpckhbw        m3, m7

    ; noise = round2(scaling[src] * grain, scaling_shift)
    pmullw           m2, m8
    pmullw           m3, m5
    pmulhrsw         m2, m11
    pmulhrsw         m3, m11

    ; dst = clip_pixel(src, noise)
    paddw            m0, m2
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    packuswb         m0, m1
    mova    [dstq+srcq], m0

    add            srcq, strideq
    add      grain_lutq, 82
    dec              hd
    jg .loop_y

    add              wq, 32
    jge .end
    lea            srcq, [src_bakq+wq]
    test       overlapd, overlapd
    jz .loop_x

    ; r8m = sbym
    movd           xm15, [pb_27_17_17_27]
    cmp       dword r8m, 0
    jne .loop_x_hv_overlap

    ; horizontal overlap (without vertical overlap)
    movd           xm14, [pw_1024]
.loop_x_h_overlap:
    mov             r6d, seed
    or             seed, 0xEFF4
    shr             r6d, 1
    test           seeb, seeh
    lea            seed, [r6+0x8000]
    cmovp          seed, r6d                ; updated seed

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                offx, offy, see, left_offxy

    lea     left_offxyd, [offyd+32]         ; previous column's offy*stride+offx
    mov           offxd, seed
    rorx          offyd, seed, 8
    shr           offxd, 12
    and           offyd, 0xf
    imul          offyd, 164
    lea           offyq, [offyq+offxq*2+747] ; offy*stride+offx

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                h, offxy, see, left_offxy

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
.loop_y_h_overlap:
    ; src
    mova             m0, [srcq]
    pxor             m2, m2
    punpckhbw        m1, m0, m2
    punpcklbw        m0, m2                 ; m0-1: src as word
    punpckhwd        m5, m0, m2
    punpcklwd        m4, m0, m2
    punpckhwd        m7, m1, m2
    punpcklwd        m6, m1, m2             ; m4-7: src as dword

    ; scaling[src]
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m8, [scalingq+m4], m3
    vpgatherdd       m4, [scalingq+m5], m9
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m5, [scalingq+m6], m3
    vpgatherdd       m6, [scalingq+m7], m9
    pand             m8, m10
    pand             m4, m10
    pand             m5, m10
    pand             m6, m10
    packusdw         m8, m4
    packusdw         m5, m6

    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq]
    movd            xm4, [grain_lutq+left_offxyq]
    punpcklbw       xm4, xm3
    pmaddubsw       xm4, xm15, xm4
    pmulhrsw        xm4, xm14
    packsswb        xm4, xm4
    vpblendw        xm4, xm3, 11111110b
    vpblendd         m3, m4, 00001111b
    pcmpgtb          m7, m2, m3
    punpcklbw        m2, m3, m7
    punpckhbw        m3, m7

    ; noise = round2(scaling[src] * grain, scaling_shift)
    pmullw           m2, m8
    pmullw           m3, m5
    pmulhrsw         m2, m11
    pmulhrsw         m3, m11

    ; dst = clip_pixel(src, noise)
    paddw            m0, m2
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    packuswb         m0, m1
    mova    [dstq+srcq], m0

    add            srcq, strideq
    add      grain_lutq, 82
    dec              hd
    jg .loop_y_h_overlap

    add              wq, 32
    jge .end
    lea            srcq, [src_bakq+wq]

    ; r8m = sbym
    cmp       dword r8m, 0
    jne .loop_x_hv_overlap
    jmp .loop_x_h_overlap

.end:
    RET

.vertical_overlap:
    DEFINE_ARGS dst, src, stride, fg_data, w, scaling, grain_lut, unused, sby, see, overlap

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

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                unused1, unused2, see, overlap

    lea        src_bakq, [srcq+wq]
    neg              wq
    sub            dstq, srcq

    vpbroadcastd    m14, [pw_1024]
.loop_x_v_overlap:
    vpbroadcastw    m15, [pb_27_17_17_27]

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
    rorx           seed, r7d, 1             ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                offx, offy, see, overlap, top_offxy

    rorx          offyd, seed, 8
    rorx          offxd, seed, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 164
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq*2+0x10001*747+32*82]

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                h, offxy, see, overlap, top_offxy

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
.loop_y_v_overlap:
    ; src
    mova             m0, [srcq]
    pxor             m2, m2
    punpckhbw        m1, m0, m2
    punpcklbw        m0, m2                 ; m0-1: src as word
    punpckhwd        m5, m0, m2
    punpcklwd        m4, m0, m2
    punpckhwd        m7, m1, m2
    punpcklwd        m6, m1, m2             ; m4-7: src as dword

    ; scaling[src]
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m8, [scalingq+m4], m3
    vpgatherdd       m4, [scalingq+m5], m9
    pcmpeqw          m3, m3
    pcmpeqw          m9, m9
    vpgatherdd       m5, [scalingq+m6], m3
    vpgatherdd       m6, [scalingq+m7], m9
    pand             m8, m10
    pand             m4, m10
    pand             m5, m10
    pand             m6, m10
    packusdw         m8, m4
    packusdw         m5, m6

    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq]
    movu             m4, [grain_lutq+top_offxyq]
    punpckhbw        m6, m4, m3
    punpcklbw        m4, m3
    pmaddubsw        m6, m15, m6
    pmaddubsw        m4, m15, m4
    pmulhrsw         m6, m14
    pmulhrsw         m4, m14
    packsswb         m3, m4, m6
    pcmpgtb          m7, m2, m3
    punpcklbw        m2, m3, m7
    punpckhbw        m3, m7

    ; noise = round2(scaling[src] * grain, scaling_shift)
    pmullw           m2, m8
    pmullw           m3, m5
    pmulhrsw         m2, m11
    pmulhrsw         m3, m11

    ; dst = clip_pixel(src, noise)
    paddw            m0, m2
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    packuswb         m0, m1
    mova    [dstq+srcq], m0

    vpbroadcastw    m15, [pb_27_17_17_27+2] ; swap weights for second v-overlap line
    add            srcq, strideq
    add      grain_lutq, 82
    dec              hw
    jz .end_y_v_overlap
    ; 2 lines get vertical overlap, then fall back to non-overlap code for
    ; remaining (up to) 30 lines
    xor              hd, 0x10000
    test             hd, 0x10000
    jnz .loop_y_v_overlap
    jmp .loop_y

.end_y_v_overlap:
    add              wq, 32
    jge .end_hv
    lea            srcq, [src_bakq+wq]

    ; since fg_dataq.overlap is guaranteed to be set, we never jump
    ; back to .loop_x_v_overlap, and instead always fall-through to
    ; h+v overlap

    movd           xm15, [pb_27_17_17_27]
.loop_x_hv_overlap:
    vpbroadcastw     m8, [pb_27_17_17_27]

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
    rorx           seed, r7d, 1             ; updated (cur_seed << 16) | top_seed

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                offx, offy, see, left_offxy, top_offxy, topleft_offxy

    lea  topleft_offxyq, [top_offxyq+32]
    lea     left_offxyq, [offyq+32]
    rorx          offyd, seed, 8
    rorx          offxd, seed, 12
    and           offyd, 0xf000f
    and           offxd, 0xf000f
    imul          offyd, 164
    ; offxy=offy*stride+offx, (cur_offxy << 16) | top_offxy
    lea           offyq, [offyq+offxq*2+0x10001*747+32*82]

    DEFINE_ARGS dst, src, stride, src_bak, w, scaling, grain_lut, \
                h, offxy, see, left_offxy, top_offxy, topleft_offxy

    movzx    top_offxyd, offxyw
    shr          offxyd, 16

    mov              hd, hm
    mov      grain_lutq, grain_lutmp
.loop_y_hv_overlap:
    ; src
    mova             m0, [srcq]
    pxor             m2, m2
    punpckhbw        m1, m0, m2
    punpcklbw        m0, m2                 ; m0-1: src as word
    punpckhwd        m5, m0, m2
    punpcklwd        m4, m0, m2
    punpckhwd        m7, m1, m2
    punpcklwd        m6, m1, m2             ; m4-7: src as dword

    ; scaling[src]
    pcmpeqw          m3, m3
; FIXME it would be nice to have another register here to do 2 vpgatherdd's in parallel
    vpgatherdd       m9, [scalingq+m4], m3
    pcmpeqw          m3, m3
    vpgatherdd       m4, [scalingq+m5], m3
    pcmpeqw          m3, m3
    vpgatherdd       m5, [scalingq+m6], m3
    pcmpeqw          m3, m3
    vpgatherdd       m6, [scalingq+m7], m3
    pand             m9, m10
    pand             m4, m10
    pand             m5, m10
    pand             m6, m10
    packusdw         m9, m4
    packusdw         m5, m6

    ; grain = grain_lut[offy+y][offx+x]
    movu             m3, [grain_lutq+offxyq]
    movu             m6, [grain_lutq+top_offxyq]
    movd            xm4, [grain_lutq+left_offxyq]
    movd            xm7, [grain_lutq+topleft_offxyq]
    ; do h interpolation first (so top | top/left -> top, left | cur -> cur)
    punpcklbw       xm4, xm3
    punpcklbw       xm7, xm6
    pmaddubsw       xm4, xm15, xm4
    pmaddubsw       xm7, xm15, xm7
    pmulhrsw        xm4, xm14
    pmulhrsw        xm7, xm14
    packsswb        xm4, xm4
    packsswb        xm7, xm7
    vpblendw        xm4, xm3, 11111110b
    vpblendw        xm7, xm6, 11111110b
    vpblendd         m3, m4, 00001111b
    vpblendd         m6, m7, 00001111b
    ; followed by v interpolation (top | cur -> cur)
    punpckhbw        m7, m6, m3
    punpcklbw        m6, m3
    pmaddubsw        m7, m8, m7
    pmaddubsw        m6, m8, m6
    pmulhrsw         m7, m14
    pmulhrsw         m6, m14
    packsswb         m3, m6, m7
    pcmpgtb          m7, m2, m3
    punpcklbw        m2, m3, m7
    punpckhbw        m3, m7

    ; noise = round2(scaling[src] * grain, scaling_shift)
    pmullw           m2, m9
    pmullw           m3, m5
    pmulhrsw         m2, m11
    pmulhrsw         m3, m11

    ; dst = clip_pixel(src, noise)
    paddw            m0, m2
    paddw            m1, m3
    pmaxsw           m0, m13
    pmaxsw           m1, m13
    pminsw           m0, m12
    pminsw           m1, m12
    packuswb         m0, m1
    mova    [dstq+srcq], m0

    vpbroadcastw     m8, [pb_27_17_17_27+2] ; swap weights for second v-overlap line
    add            srcq, strideq
    add      grain_lutq, 82
    dec              hw
    jz .end_y_hv_overlap
    ; 2 lines get vertical overlap, then fall back to non-overlap code for
    ; remaining (up to) 30 lines
    xor              hd, 0x10000
    test             hd, 0x10000
    jnz .loop_y_hv_overlap
    jmp .loop_y_h_overlap

.end_y_hv_overlap:
    add              wq, 32
    lea            srcq, [src_bakq+wq]
    jl .loop_x_hv_overlap

.end_hv:
    RET

%endif ; ARCH_X86_64
