; TODO: License
%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_XMM ssse3
cglobal sad_4x4, 4, 6, 8, src, src_stride, dst, dst_stride, \
                          src_stride3, dst_stride3
    lea       src_stride3q, [src_strideq*3]
    lea       dst_stride3q, [dst_strideq*3]
    movq                m0, [srcq]
    movq                m1, [srcq+src_strideq*1]
    movq                m2, [srcq+src_strideq*2]
    movq                m3, [srcq+src_stride3q]
    movq                m4, [dstq]
    movq                m5, [dstq+dst_strideq*1]
    movq                m6, [dstq+dst_strideq*2]
    movq                m7, [dstq+dst_stride3q]
    psubw               m0, m4
    psubw               m1, m5
    psubw               m2, m6
    psubw               m3, m7
    pabsw               m0, m0
    pabsw               m1, m1
    pabsw               m2, m2
    pabsw               m3, m3
    paddw               m0, m1
    paddw               m2, m3
    paddw               m0, m2
    pshuflw             m1, m0, q2323
    paddw               m0, m1
    pshuflw             m1, m0, q1111
    paddw               m0, m1
    movd               eax, m0
    movzx              eax, ax
    RET

%if ARCH_X86_64

; this should be a 10-bit version
; 10-bit only
INIT_XMM ssse3
cglobal sad_8x8, 4, 7, 9, src, src_stride, dst, dst_stride, \
                          src_stride3, dst_stride3, cnt
    lea       src_stride3q, [src_strideq*3]
    lea       dst_stride3q, [dst_strideq*3]
    mov               cntd, 2
    pxor                m0, m0
.loop:
    movu                m1, [srcq]
    movu                m2, [srcq+src_strideq*1]
    movu                m3, [srcq+src_strideq*2]
    movu                m4, [srcq+src_stride3q]
    lea               srcq, [srcq+src_strideq*4]
    movu                m5, [dstq]
    movu                m6, [dstq+dst_strideq*1]
    movu                m7, [dstq+dst_strideq*2]
    movu                m8, [dstq+dst_stride3q]
    lea               dstq, [dstq+dst_strideq*4]
    psubw               m1, m5
    psubw               m2, m6
    psubw               m3, m7
    psubw               m4, m8
    pabsw               m1, m1
    pabsw               m2, m2
    pabsw               m3, m3
    pabsw               m4, m4
    paddw               m1, m2
    paddw               m3, m4
    paddw               m0, m1
    paddw               m0, m3
    dec               cntd
    jg .loop
    movhlps             m1, m0
    paddw               m0, m1
    pshuflw             m1, m0, q2323
    paddw               m0, m1
    pshuflw             m1, m0, q1111
    paddw               m0, m1
    movd               eax, m0
    movzx              eax, ax
    RET

INIT_XMM ssse3
cglobal sad_16x16, 4, 5, 9, src, src_stride, dst, dst_stride, \
                            cnt
    mov               cntd, 8
    pxor                m0, m0
.loop:
    movu                m1, [srcq]
    movu                m2, [srcq+16]
    movu                m3, [srcq+src_strideq]
    movu                m4, [srcq+src_strideq+16]
    lea               srcq, [srcq+src_strideq*2]
    movu                m5, [dstq]
    movu                m6, [dstq+16]
    movu                m7, [dstq+dst_strideq]
    movu                m8, [dstq+dst_strideq+16]
    lea               dstq, [dstq+dst_strideq*2]
    psubw               m1, m5
    psubw               m2, m6
    psubw               m3, m7
    psubw               m4, m8
    pabsw               m1, m1
    pabsw               m2, m2
    pabsw               m3, m3
    pabsw               m4, m4
    paddw               m1, m2
    paddw               m3, m4
    paddw               m0, m1
    paddw               m0, m3
    dec               cntd
    jg .loop
; convert to 32-bit
    pxor                m1, m1
    punpcklwd           m2, m0, m1
    punpckhwd           m0, m1
    paddd               m0, m2
    movhlps             m1, m0
    paddd               m0, m1
    pshufd              m1, m0, q1111
    paddd               m0, m1
    movd               eax, m0
    RET

;10 bit only
INIT_XMM ssse3
cglobal sad_32x32, 4, 5, 10, src, src_stride, dst, dst_stride, \
                            cnt
    mov               cntd, 32
    pxor                m0, m0
    pxor                m9, m9
.loop:
    movu                m1, [srcq]
    movu                m2, [srcq+16]
    movu                m3, [srcq+32]
    movu                m4, [srcq+48]
    lea               srcq, [srcq+src_strideq]
    movu                m5, [dstq]
    movu                m6, [dstq+16]
    movu                m7, [dstq+32]
    movu                m8, [dstq+48]
    lea               dstq, [dstq+dst_strideq]
    psubw               m1, m5
    psubw               m2, m6
    psubw               m3, m7
    psubw               m4, m8
    pabsw               m1, m1
    pabsw               m2, m2
    pabsw               m3, m3
    pabsw               m4, m4
    paddw               m1, m2
    paddw               m3, m4
    paddw               m0, m1
    paddw               m9, m3
    dec               cntd
    jg .loop
; convert to 32-bit
    pxor                m1, m1
    punpcklwd           m2, m0, m1
    punpckhwd           m0, m1
    paddd               m0, m2
    punpcklwd           m3, m9, m1
    punpckhwd           m9, m1
    paddd               m9, m3
    paddd               m0, m9
    movhlps             m1, m0
    paddd               m0, m1
    pshufd              m1, m0, q1111
    paddd               m0, m1
    movd               eax, m0
    RET

%endif
