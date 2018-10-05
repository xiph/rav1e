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

SECTION_RODATA 32

pb_4x1_4x5_4x9_4x13: times 2 db 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12
pb_7_1: times 16 db 7, 1
pb_3_1: times 16 db 3, 1
pb_2_1: times 16 db 2, 1
pb_m1_0: times 16 db -1, 0
pb_m1_1: times 16 db -1, 1
pb_m1_2: times 16 db -1, 2
pb_1: times 32 db 1
pb_2: times 32 db 2
pb_3: times 32 db 3
pb_4: times 32 db 4
pb_16: times 32 db 16
pb_63: times 32 db 63
pb_64: times 32 db 64
pb_128: times 32 db 0x80
pb_129: times 32 db 0x81
pb_240: times 32 db 0xf0
pb_248: times 32 db 0xf8
pb_254: times 32 db 0xfe

pw_2048: times 16 dw 2048
pw_4096: times 16 dw 4096

pb_mask: dd 1, 2, 4, 8, 16, 32, 64, 128

SECTION .text

%macro ABSSUB 4 ; dst, a, b, tmp
    psubusb       %1, %2, %3
    psubusb       %4, %3, %2
    por           %1, %4
%endmacro

%macro FILTER 1 ; width
    movu          m1, [lq]
    movu          m0, [lq+l_strideq]
    pxor          m2, m2
    pcmpeqb       m3, m2, m0
    pand          m1, m3
    por           m0, m1                        ; l[x][] ? l[x][] : l[x-stride][]
    pshufb        m0, [pb_4x1_4x5_4x9_4x13]     ; l[x][1]
    pcmpeqb      m10, m2, m0                    ; !L
    pand          m1, m0, [pb_240]
    psrlq         m1, 4                         ; H
    psrlq         m2, m0, [lutq+128]
    pand          m2, [pb_63]
    vpbroadcastb  m4, [lutq+136]
    pminub        m2, m4
    pmaxub        m2, [pb_1]                    ; I
    paddb         m0, [pb_2]
    paddb         m0, m0
    paddb         m0, m2                        ; E
    pxor          m1, [pb_128]
    pxor          m2, [pb_128]
    pxor          m0, [pb_128]

%if %1 == 4
    lea         tmpq, [dstq+mstrideq*2]
    mova          m3, [tmpq+strideq*0]          ; p1
    mova          m4, [tmpq+strideq*1]          ; p0
    mova          m5, [tmpq+strideq*2]          ; q0
    mova          m6, [tmpq+stride3q]           ; q1
%else
    lea         tmpq, [dstq+mstrideq*4]
    mova          m3, [tmpq+strideq*2]
    mova          m4, [tmpq+stride3q]
    mova          m5, [dstq+strideq*0]
    mova          m6, [dstq+strideq*1]
%endif

    ABSSUB        m8, m3, m4, m9                ; abs(p1-p0)
    pmaxub        m8, m10
    ABSSUB        m9, m5, m6, m10               ; abs(q1-q0)
    pmaxub        m8, m9
%if %1 == 4
    pxor          m8, [pb_128]
    pcmpgtb       m7, m8, m1                    ; hev
%else
    pxor          m7, m8, [pb_128]
    pcmpgtb       m7, m1                        ; hev

%if %1 != 6
    mova         m12, [tmpq+strideq*0]
%endif
    mova         m13, [tmpq+strideq*1]
    mova         m14, [dstq+strideq*2]
%if %1 != 6
    mova         m15, [dstq+stride3q]
%endif

%if %1 == 6
    ABSSUB        m9, m13, m4, m10              ; abs(p2-p0)
    pmaxub        m9, m8
%else
    ABSSUB        m9, m12, m4, m10              ; abs(p3-p0)
    pmaxub        m9, m8
    ABSSUB       m10, m13, m4, m11              ; abs(p2-p0)
    pmaxub        m9, m10
%endif
    ABSSUB       m10, m5,  m14, m11             ; abs(q2-q0)
    pmaxub        m9, m10
%if %1 != 6
    ABSSUB       m10, m5,  m15, m11             ; abs(q3-q0)
    pmaxub        m9, m10
%endif
    pxor          m9, [pb_128]
    pcmpgtb       m9, [pb_129]                  ; !flat8in

%if %1 == 6
    ABSSUB       m10, m13, m3,  m1              ; abs(p2-p1)
%else
    ABSSUB       m10, m12, m13, m11             ; abs(p3-p2)
    ABSSUB       m11, m13, m3,  m1              ; abs(p2-p1)
    pmaxub       m10, m11
    ABSSUB       m11, m14, m15, m1              ; abs(q3-q2)
    pmaxub       m10, m11
%endif
    ABSSUB       m11, m14, m6,  m1              ; abs(q2-q1)
    pmaxub       m10, m11
%if %1 == 16
    vpbroadcastd m11, [maskq+8]
    vpbroadcastd  m1, [maskq+4]
    por          m11, m1
    pand         m11, [pb_mask]
    pcmpeqd      m11, [pb_mask]
    pand         m10, m11
%else
    vpbroadcastd m11, [maskq+4]
    pand         m11, [pb_mask]
    pcmpeqd      m11, [pb_mask]
    pand         m10, m11                       ; only apply fm-wide to wd>4 blocks
%endif
    pmaxub        m8, m10

    pxor          m8, [pb_128]
%endif
    pcmpgtb       m8, m2

    ABSSUB       m10, m3, m6, m11               ; abs(p1-q1)
    ABSSUB       m11, m4, m5, m2                ; abs(p0-q0)
    paddusb      m11, m11
    pand         m10, [pb_254]
    psrlq        m10, 1
    paddusb      m10, m11                       ; abs(p0-q0)*2+(abs(p1-q1)>>1)
    pxor         m10, [pb_128]
    pcmpgtb      m10, m0                        ; abs(p0-q0)*2+(abs(p1-q1)>>1) > E
    por           m8, m10

%if %1 == 16
    lea         tmpq, [dstq+mstrideq*8]
    mova          m0, [tmpq+strideq*1]
    ABSSUB        m1, m0, m4, m2
    mova          m0, [tmpq+strideq*2]
    ABSSUB        m2, m0, m4, m10
    pmaxub        m1, m2
    mova          m0, [tmpq+stride3q]
    ABSSUB        m2, m0, m4, m10
    pmaxub        m1, m2
    lea         tmpq, [dstq+strideq*4]
    mova          m0, [tmpq+strideq*0]
    ABSSUB        m2, m0, m5, m10
    pmaxub        m1, m2
    mova          m0, [tmpq+strideq*1]
    ABSSUB        m2, m0, m5, m10
    pmaxub        m1, m2
    mova          m0, [tmpq+strideq*2]
    ABSSUB        m2, m0, m5, m10
    pmaxub        m1, m2
    pxor          m1, [pb_128]
    pcmpgtb       m1, [pb_129]                  ; !flat8out
    por           m1, m9                        ; !flat8in | !flat8out
    vpbroadcastd  m2, [maskq+8]
    pand         m10, m2, [pb_mask]
    pcmpeqd      m10, [pb_mask]
    pandn         m1, m10                       ; flat16
    pandn         m1, m8, m1                    ; flat16 & fm

    vpbroadcastd m10, [maskq+4]
    por          m10, m2
    pand          m2, m10, [pb_mask]
    pcmpeqd       m2, [pb_mask]
    pandn         m9, m2                        ; flat8in
    pandn         m9, m8, m9
    vpbroadcastd  m2, [maskq+0]
    por           m2, m10
    pand          m2, [pb_mask]
    pcmpeqd       m2, [pb_mask]
    pandn         m8, m2
    pandn         m8, m9, m8                    ; fm & !flat8 & !flat16
    pandn         m9, m1, m9                    ; flat8 & !flat16
%elif %1 != 4
    vpbroadcastd  m0, [maskq+4]
    pand          m2, m0, [pb_mask]
    pcmpeqd       m2, [pb_mask]
    pandn         m9, m2
    pandn         m9, m8, m9                    ; flat8 & fm
    vpbroadcastd  m2, [maskq+0]
    por           m0, m2
    pand          m0, [pb_mask]
    pcmpeqd       m0, [pb_mask]
    pandn         m8, m0
    pandn         m8, m9, m8                    ; fm & !flat8
%else
    vpbroadcastd  m0, [maskq+0]
    pand          m0, [pb_mask]
    pcmpeqd       m0, [pb_mask]
    pandn         m8, m0                        ; fm
%endif

    ; short filter

    pxor          m3, [pb_128]
    pxor          m6, [pb_128]
    psubsb       m10, m3, m6                    ; iclip_diff(p1-q1)
    pand         m10, m7                        ; f=iclip_diff(p1-q1)&hev
    pxor          m4, [pb_128]
    pxor          m5, [pb_128]
    psubsb       m11, m5, m4
    paddsb       m10, m11
    paddsb       m10, m11
    paddsb       m10, m11                       ; f=iclip_diff(3*(q0-p0)+f)
    pand          m8, m10                       ; f&=fm
    paddsb       m10, m8, [pb_3]
    paddsb        m8, [pb_4]
    pand         m10, [pb_248]
    pand          m8, [pb_248]
    psrlq        m10, 3
    psrlq         m8, 3
    pxor         m10, [pb_16]
    pxor          m8, [pb_16]
    psubb        m10, [pb_16]                   ; f2
    psubb         m8, [pb_16]                   ; f1
    paddsb        m4, m10
    psubsb        m5, m8
    pxor          m4, [pb_128]
    pxor          m5, [pb_128]

    pxor          m8, [pb_128]
    pxor         m10, m10
    pavgb         m8, m10                       ; f=(f1+1)>>1
    psubb         m8, [pb_64]
    pandn         m8, m7, m8                    ; f&=!hev
    paddsb        m3, m8
    psubsb        m6, m8
    pxor          m3, [pb_128]
    pxor          m6, [pb_128]

%if %1 == 16
    ; flat16 filter
    lea         tmpq, [dstq+mstrideq*8]
    mova          m0, [tmpq+strideq*1]          ; p6
    mova          m2, [tmpq+strideq*2]          ; p5
    mova          m7, [tmpq+stride3q]           ; p4

    mova  [rsp+0*32], m9
    mova  [rsp+1*32], m14
    mova  [rsp+2*32], m15

    ; p6*7+p5*2+p4*2+p3+p2+p1+p0+q0 [p5/p4/p2/p1/p0/q0][p6/p3] A
    ; write -6
    punpcklbw    m14, m0, m12
    punpckhbw    m15, m0, m12
    pmaddubsw    m10, m14, [pb_7_1]
    pmaddubsw    m11, m15, [pb_7_1]             ; p6*7+p3
    punpcklbw     m8, m2, m7
    punpckhbw     m9, m2, m7
    pmaddubsw     m8, [pb_2]
    pmaddubsw     m9, [pb_2]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*7+p5*2+p4*2+p3
    punpcklbw     m8, m13, m3
    punpckhbw     m9, m13, m3
    pmaddubsw     m8, [pb_1]
    pmaddubsw     m9, [pb_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*7+p5*2+p4*2+p3+p2+p1
    punpcklbw     m8, m4, m5
    punpckhbw     m9, m4, m5
    pmaddubsw     m8, [pb_1]
    pmaddubsw     m9, [pb_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*7+p5*2+p4*2+p3+p2+p1+p0+q0
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m2
    por           m8, m9
    mova [tmpq+strideq*2], m8                   ; p5

    ; sub p6*2, add p3/q1 [reuse p6/p3 from A][-p6,+q1|save] B
    ; write -5
    pmaddubsw    m14, [pb_m1_1]
    pmaddubsw    m15, [pb_m1_1]
    paddw        m10, m14
    paddw        m11, m15                       ; p6*6+p5*2+p4*2+p3*2+p2+p1+p0+q0
    punpcklbw     m8, m0, m6
    punpckhbw     m9, m0, m6
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    mova  [rsp+3*32], m8
    mova  [rsp+4*32], m9
    paddw        m10, m8
    paddw        m11, m9                        ; p6*5+p5*2+p4*2+p3*2+p2+p1+p0+q0+q1
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m7
    por           m8, m9
    mova [tmpq+stride3q], m8                    ; p4

    ; sub p6/p5, add p2/q2 [-p6,+p2][-p5,+q2|save] C
    ; write -4
    mova         m14, [rsp+1*32]
    punpcklbw     m8, m0, m13
    punpckhbw     m9, m0, m13
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*4+p5*2+p4*2+p3*2+p2*2+p1+p0+q0+q1
    punpcklbw     m8, m2, m14
    punpckhbw     m2, m14
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m2, [pb_m1_1]
    mova  [rsp+1*32], m8
    paddw        m10, m8
    paddw        m11, m2                        ; p6*4+p5+p4*2+p3*2+p2*2+p1+p0+q0+q1+q2
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m12
    por           m8, m9
    mova [tmpq+strideq*4], m8                   ; p3

    ; sub p6/p4, add p1/q3 [-p6,+p1][-p4,+q3|save] D
    ; write -3
    mova         m15, [rsp+2*32]
    punpcklbw     m8, m0, m3
    punpckhbw     m9, m0, m3
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*3+p5+p4*2+p3*2+p2*2+p1*2+p0+q0+q1+q2
    punpcklbw     m8, m7, m15
    punpckhbw     m7, m15
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m7, [pb_m1_1]
    mova  [rsp+2*32], m8
    paddw        m10, m8
    paddw        m11, m7                        ; p6*3+p5+p4+p3*2+p2*2+p1*2+p0+q0+q1+q2+q3
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m13
    por           m8, m9
    mova  [rsp+6*32], m8                        ; don't clobber p2/m13 since we need it in F

    ; sub p6/p3, add p0/q4 [-p6,+p0][-p3,+q4|save] E
    ; write -2
    lea         tmpq, [dstq+strideq*4]
    punpcklbw     m8, m0, m4
    punpckhbw     m9, m0, m4
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p6*2+p5+p4+p3*2+p2*2+p1*2+p0*2+q0+q1+q2+q3
    mova          m9, [tmpq+strideq*0]          ; q4
    punpcklbw     m8, m12, m9
    punpckhbw     m9, m12, m9
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    mova  [rsp+7*32], m8
    mova  [rsp+5*32], m9
    paddw        m10, m8
    paddw        m11, m9                        ; p6*2+p5+p4+p3+p2*2+p1*2+p0*2+q0+q1+q2+q3+q4
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m3
    por           m8, m9
    mova  [rsp+8*32], m8                        ; don't clobber p1/m3 since we need it in G

    ; sub p6/p2, add q0/q5 [-p6,+q0][-p2,+q5|save] F
    ; write -1
    mova          m9, [tmpq+strideq*1]          ; q5
    punpcklbw     m8, m0, m5
    punpckhbw     m0, m5
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m0, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m0                        ; p6+p5+p4+p3+p2*2+p1*2+p0*2+q0*2+q1+q2+q3+q4
    punpcklbw     m0, m13, m9
    punpckhbw     m9, m13, m9
    mova         m13, [rsp+6*32]
    pmaddubsw     m0, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    mova [rsp+ 9*32], m0
    mova [rsp+10*32], m9
    paddw        m10, m0
    paddw        m11, m9                        ; p6+p5+p4+p3+p2+p1*2+p0*2+q0*2+q1+q2+q3+q4+q5
    pmulhrsw      m0, m10, [pw_2048]
    pmulhrsw      m8, m11, [pw_2048]
    packuswb      m0, m8
    pand          m0, m1
    pandn         m8, m1, m4
    por           m0, m8
    mova  [rsp+6*32], m0                        ; don't clobber p0/m4 since we need it in H

    ; sub p6/p1, add q1/q6 [reuse -p6,+q1 from B][-p1,+q6|save] G
    ; write +0
    mova          m0, [tmpq+strideq*2]          ; q6
    paddw        m10, [rsp+3*32]
    paddw        m11, [rsp+4*32]                ; p5+p4+p3+p2+p1*2+p0*2+q0*2+q1*2+q2+q3+q4+q5
    punpcklbw     m8, m3, m0
    punpckhbw     m9, m3, m0
    mova          m3, [rsp+8*32]
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    mova  [rsp+3*32], m8
    mova  [rsp+4*32], m9
    paddw        m10, m8
    paddw        m11, m9                        ; p5+p4+p3+p2+p1+p0*2+q0*2+q1*2+q2+q3+q4+q5+q6
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m5
    por           m8, m9
    mova  [rsp+8*32], m8                        ; don't clobber q0/m5 since we need it in I

    ; sub p5/p0, add q2/q6 [reuse -p5,+q2 from C][-p0,+q6] H
    ; write +1
    paddw        m10, [rsp+1*32]
    paddw        m11, m2                        ; p4+p3+p2+p1+p0*2+q0*2+q1*2+q2*2+q3+q4+q5+q6
    punpcklbw     m8, m4, m0
    punpckhbw     m2, m4, m0
    mova          m4, [rsp+6*32]
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m2, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m2                        ; p4+p3+p2+p1+p0+q0*2+q1*2+q2*2+q3+q4+q5+q6*2
    pmulhrsw      m2, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m2, m9
    pand          m2, m1
    pandn         m9, m1, m6
    por           m2, m9                        ; don't clobber q1/m6 since we need it in K

    ; sub p4/q0, add q3/q6 [reuse -p4,+q3 from D][-q0,+q6] I
    ; write +2
    paddw        m10, [rsp+2*32]
    paddw        m11, m7                        ; p3+p2+p1+p0+q0*2+q1*2+q2*2+q3*2+q4+q5+q6*2
    punpcklbw     m8, m5, m0
    punpckhbw     m9, m5, m0
    mova          m5, [rsp+8*32]
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p3+p2+p1+p0+q0+q1*2+q2*2+q3*2+q4+q5+q6*3
    pmulhrsw      m7, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m7, m9
    pand          m7, m1
    pandn         m9, m1, m14
    por           m7, m9                        ; don't clobber q2/m14 since we need it in K

    ; sub p3/q1, add q4/q6 [reuse -p3,+q4 from E][-q1,+q6] J
    ; write +3
    paddw        m10, [rsp+7*32]
    paddw        m11, [rsp+5*32]                ; p2+p1+p0+q0+q1*2+q2*2+q3*2+q4*2+q5+q6*3
    punpcklbw     m8, m6, m0
    punpckhbw     m9, m6, m0
    SWAP           2, 6
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p2+p1+p0+q0+q1+q2*2+q3*2+q4*2+q5+q6*4
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, m15
    por           m8, m9
    mova [tmpq+mstrideq], m8                    ; q3

    ; sub p2/q2, add q5/q6 [reuse -p2,+q5 from F][-q2,+q6] K
    ; write +4
    paddw        m10, [rsp+ 9*32]
    paddw        m11, [rsp+10*32]               ; p1+p0+q0+q1+q2*2+q3*2+q4*2+q5*2+q6*4
    punpcklbw     m8, m14, m0
    punpckhbw     m9, m14, m0
    SWAP          14, 7
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p1+p0+q0+q1+q2+q3*2+q4*2+q5*2+q6*5
    pmulhrsw      m8, m10, [pw_2048]
    pmulhrsw      m9, m11, [pw_2048]
    packuswb      m8, m9
    pand          m8, m1
    pandn         m9, m1, [tmpq+strideq*0]
    por           m8, m9
    mova [tmpq+strideq*0], m8                    ; q4

    ; sub p1/q3, add q6*2 [reuse -p1,+q6 from G][-q3,+q6] L
    ; write +5
    paddw        m10, [rsp+3*32]
    paddw        m11, [rsp+4*32]                ; p1+p0+q0+q1+q2*2+q3*2+q4*2+q5*2+q6*4
    punpcklbw     m8, m15, m0
    punpckhbw     m9, m15, m0
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw     m9, [pb_m1_1]
    paddw        m10, m8
    paddw        m11, m9                        ; p1+p0+q0+q1+q2+q3*2+q4*2+q5*2+q6*5
    pmulhrsw     m10, [pw_2048]
    pmulhrsw     m11, [pw_2048]
    packuswb     m10, m11
    pand         m10, m1
    pandn        m11, m1, [tmpq+strideq*1]
    por          m10, m11
    mova [tmpq+strideq*1], m10                  ; q5

    mova          m9, [rsp+0*32]
    lea         tmpq, [dstq+mstrideq*4]
%endif
%if %1 >= 8
    ; flat8 filter

    punpcklbw     m0, m12, m3
    punpckhbw     m1, m12, m3
    pmaddubsw     m2, m0, [pb_3_1]
    pmaddubsw     m7, m1, [pb_3_1]              ; 3 * p3 + p1
    punpcklbw     m8, m13, m4
    punpckhbw    m11, m13, m4
    pmaddubsw     m8, [pb_2_1]
    pmaddubsw    m11, [pb_2_1]
    paddw         m2, m8
    paddw         m7, m11                       ; 3 * p3 + 2 * p2 + p1 + p0
    punpcklbw     m8, m5, [pb_4]
    punpckhbw    m11, m5, [pb_4]
    pmaddubsw     m8, [pb_1]
    pmaddubsw    m11, [pb_1]
    paddw         m2, m8
    paddw         m7, m11                       ; 3 * p3 + 2 * p2 + p1 + p0 + q0 + 4
    psrlw         m8, m2, 3
    psrlw        m11, m7, 3
    packuswb      m8, m11
    pand          m8, m9
    pandn        m11, m9, m13
    por           m8, m11                      ; p2
    mova [tmpq+strideq*1], m8                  ; p2

    pmaddubsw     m8, m0, [pb_m1_1]
    pmaddubsw    m11, m1, [pb_m1_1]
    paddw         m2, m8
    paddw         m7, m11
    punpcklbw     m8, m13, m6
    punpckhbw    m11, m13, m6
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw    m11, [pb_m1_1]
    paddw         m2, m8
    paddw         m7, m11                       ; 2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4
    psrlw         m8, m2, 3
    psrlw        m11, m7, 3
    packuswb      m8, m11
    pand          m8, m9
    pandn        m11, m9, m3
    por           m8, m11                       ; p1
    mova [tmpq+strideq*2], m8                   ; p1

    pmaddubsw     m0, [pb_1]
    pmaddubsw     m1, [pb_1]
    psubw         m2, m0
    psubw         m7, m1
    punpcklbw     m8, m4, m14
    punpckhbw    m11, m4, m14
    pmaddubsw     m8, [pb_1]
    pmaddubsw    m11, [pb_1]
    paddw         m2, m8
    paddw         m7, m11                       ; p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4
    psrlw         m8, m2, 3
    psrlw        m11, m7, 3
    packuswb      m8, m11
    pand          m8, m9
    pandn        m11, m9, m4
    por           m8, m11                       ; p0
    mova [tmpq+stride3q ], m8                   ; p0

    punpcklbw     m0, m5, m15
    punpckhbw     m1, m5, m15
    pmaddubsw     m8, m0, [pb_1]
    pmaddubsw    m11, m1, [pb_1]
    paddw         m2, m8
    paddw         m7, m11
    punpcklbw     m8, m4, m12
    punpckhbw    m11, m4, m12
    pmaddubsw     m8, [pb_1]
    pmaddubsw    m11, [pb_1]
    psubw         m2, m8
    psubw         m7, m11                       ; p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4
    psrlw         m8, m2, 3
    psrlw        m11, m7, 3
    packuswb      m8, m11
    pand          m8, m9
    pandn        m11, m9, m5
    por           m8, m11                       ; q0
    mova [dstq+strideq*0], m8                   ; q0

    pmaddubsw     m0, [pb_m1_1]
    pmaddubsw     m1, [pb_m1_1]
    paddw         m2, m0
    paddw         m7, m1
    punpcklbw     m8, m13, m6
    punpckhbw    m11, m13, m6
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw    m11, [pb_m1_1]
    paddw         m2, m8
    paddw         m7, m11                       ; p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4
    psrlw         m8, m2, 3
    psrlw        m11, m7, 3
    packuswb      m8, m11
    pand          m8, m9
    pandn        m11, m9, m6
    por           m8, m11                       ; q1
    mova [dstq+strideq*1], m8                   ; q1

    punpcklbw     m0, m3, m6
    punpckhbw     m1, m3, m6
    pmaddubsw     m0, [pb_1]
    pmaddubsw     m1, [pb_1]
    psubw         m2, m0
    psubw         m7, m1
    punpcklbw     m0, m14, m15
    punpckhbw     m1, m14, m15
    pmaddubsw     m0, [pb_1]
    pmaddubsw     m1, [pb_1]
    paddw         m2, m0
    paddw         m7, m1                        ; p0 + q0 + q1 + q2 + 2 * q2 + 3 * q3 + 4
    psrlw         m2, 3
    psrlw         m7, 3
    packuswb      m2, m7
    pand          m2, m9
    pandn        m11, m9, m14
    por           m2, m11                       ; q2
    mova [dstq+strideq*2], m2                   ; q2
%elif %1 == 6
    ; flat6 filter

    punpcklbw     m8, m13, m5
    punpckhbw    m11, m13, m5
    pmaddubsw     m0, m8, [pb_3_1]
    pmaddubsw     m1, m11, [pb_3_1]
    punpcklbw     m7, m4, m3
    punpckhbw    m10, m4, m3
    pmaddubsw     m2, m7, [pb_2]
    pmaddubsw    m12, m10, [pb_2]
    paddw         m0, m2
    paddw         m1, m12
    pmulhrsw      m2, m0, [pw_4096]
    pmulhrsw     m12, m1, [pw_4096]
    packuswb      m2, m12
    pand          m2, m9
    pandn        m12, m9, m3
    por           m2, m12
    mova [tmpq+strideq*2], m2                   ; p1

    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw    m11, [pb_m1_1]
    paddw         m0, m8
    paddw         m1, m11
    punpcklbw     m8, m13, m6
    punpckhbw    m11, m13, m6
    pmaddubsw     m8, [pb_m1_1]
    pmaddubsw    m11, [pb_m1_1]
    paddw         m0, m8
    paddw         m1, m11
    pmulhrsw     m12, m0, [pw_4096]
    pmulhrsw     m13, m1, [pw_4096]
    packuswb     m12, m13
    pand         m12, m9
    pandn        m13, m9, m4
    por          m12, m13
    mova [tmpq+stride3q], m12                   ; p0

    paddw         m0, m8
    paddw         m1, m11
    punpcklbw     m8, m3, m14
    punpckhbw    m11, m3, m14
    pmaddubsw    m12, m8, [pb_m1_1]
    pmaddubsw    m13, m11, [pb_m1_1]
    paddw         m0, m12
    paddw         m1, m13
    pmulhrsw     m12, m0, [pw_4096]
    pmulhrsw     m13, m1, [pw_4096]
    packuswb     m12, m13
    pand         m12, m9
    pandn        m13, m9, m5
    por          m12, m13
    mova [dstq+strideq*0], m12                  ; q0

    pmaddubsw     m8, [pb_m1_2]
    pmaddubsw    m11, [pb_m1_2]
    paddw         m0, m8
    paddw         m1, m11
    pmaddubsw     m7, [pb_m1_0]
    pmaddubsw    m10, [pb_m1_0]
    paddw         m0, m7
    paddw         m1, m10
    pmulhrsw      m0, [pw_4096]
    pmulhrsw      m1, [pw_4096]
    packuswb      m0, m1
    pand          m0, m9
    pandn         m9, m6
    por           m0, m9
    mova [dstq+strideq*1], m0                   ; q1
%else
    mova [tmpq+strideq*0], m3                   ; p1
    mova [tmpq+strideq*1], m4                   ; p0
    mova [tmpq+strideq*2], m5                   ; q0
    mova [tmpq+stride3q ], m6                   ; q1
%endif
%endmacro

INIT_YMM avx2
cglobal lpf_v_sb128y, 7, 10, 16, 32 * 11, \
                      dst, stride, mask, l, l_stride, lut, \
                      w, stride3, mstride, tmp
    shl    l_strideq, 2
    sub           lq, l_strideq
    mov     mstrideq, strideq
    neg     mstrideq
    lea     stride3q, [strideq*3]

.loop:
    cmp byte [maskq+8], 0                       ; vmask[2]
    je .no_flat16

    FILTER        16
    jmp .end

.no_flat16:
    cmp byte [maskq+4], 0                       ; vmask[1]
    je .no_flat

    FILTER         8
    jmp .end

.no_flat:
    cmp byte [maskq+0], 0                       ; vmask[0]
    je .end

    FILTER         4

.end:
    add           lq, 32
    add         dstq, 32
    add        maskq, 1
    sub           wd, 8
    jg .loop
    RET

INIT_YMM avx2
cglobal lpf_v_sb128uv, 7, 10, 16, \
                       dst, stride, mask, l, l_stride, lut, \
                       w, stride3, mstride, tmp
    shl    l_strideq, 2
    sub           lq, l_strideq
    mov     mstrideq, strideq
    neg     mstrideq
    lea     stride3q, [strideq*3]

.loop:
    cmp byte [maskq+4], 0                       ; vmask[1]
    je .no_flat

    FILTER         6
    jmp .end

.no_flat:
    cmp byte [maskq+0], 0                       ; vmask[0]
    je .end

    FILTER         4

.end:
    add           lq, 32
    add         dstq, 32
    add        maskq, 1
    sub           wd, 8
    jg .loop
    RET

%endif ; ARCH_X86_64
