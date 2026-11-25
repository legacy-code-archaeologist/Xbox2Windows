; coverage: record cpu_aot_sha256
; Source-owned differential fixture for Intel SHA-256 NI instructions:
; SHA256RNDS2 (implicit XMM0), SHA256MSG1, and SHA256MSG2 across register,
; flat [rdx]/[r8] memory operands, aliasing forms (xmm1==xmm2, xmm1==xmm0,
; xmm2==xmm0, all xmm0), a multi-instruction pipeline, and a discarded faulting read.
;   RCX = const void* input_a (16 bytes; 16B-aligned for movdqa / sha256* m128)
;   RDX = const void* input_b (16 bytes; 16B-aligned for movdqa / sha256* m128)
;   R8  = void*       output / in-out wk (16 bytes; 16B-aligned for movdqa / sha256* m128)
.code

; --- SHA256RNDS2 (uses implicit XMM0[63:0] as WK1:WK0) ---

; rcx = dst (CDGH), rdx = src (ABEF), [r8] in = XMM0 (WK1:WK0), [r8] out = new ABEF.
op_sha256rnds2_reg PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    movdqu xmm0, xmmword ptr [r8]
IFDEF ALTER_SHA256
    sha256rnds2 xmm2, xmm1, xmm0
    movdqu xmm1, xmm2
ELSE
    sha256rnds2 xmm1, xmm2, xmm0
ENDIF
    movdqu xmmword ptr [r8], xmm1
    ret
op_sha256rnds2_reg ENDP

op_sha256rnds2_mem PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
    movdqa xmm0, xmmword ptr [r8]
IFDEF ALTER_SHA256
    sha256rnds2 xmm0, xmmword ptr [rdx], xmm0
    movdqa xmm1, xmm0
ELSE
    sha256rnds2 xmm1, xmmword ptr [rdx], xmm0
ENDIF
    movdqa xmmword ptr [r8], xmm1
    ret
op_sha256rnds2_mem ENDP

op_sha256rnds2_mem_r8 PROC FRAME
    .endprolog
    movdqa xmm3, xmmword ptr [rcx]
    movdqa xmm0, xmmword ptr [rdx]
    sha256rnds2 xmm3, xmmword ptr [r8], xmm0
    movdqa xmmword ptr [r8], xmm3
    ret
op_sha256rnds2_mem_r8 ENDP

; Aliasing: dst == src (xmm1 == xmm2) with xmm0 from [rdx].
op_sha256rnds2_self_src PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm0, xmmword ptr [rdx]
    sha256rnds2 xmm1, xmm1, xmm0
    movdqu xmmword ptr [r8], xmm1
    ret
op_sha256rnds2_self_src ENDP

; Aliasing: dst == implicit xmm0 (xmm1 == xmm0) with src from [rdx].
op_sha256rnds2_dst_xmm0 PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    sha256rnds2 xmm0, xmm2, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha256rnds2_dst_xmm0 ENDP

; Aliasing: src == implicit xmm0 (xmm2 == xmm0) with dst from [rcx].
op_sha256rnds2_src_xmm0 PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm0, xmmword ptr [rdx]
    sha256rnds2 xmm1, xmm0, xmm0
    movdqu xmmword ptr [r8], xmm1
    ret
op_sha256rnds2_src_xmm0 ENDP

; Aliasing: dst == src == implicit xmm0.
op_sha256rnds2_all_xmm0 PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    sha256rnds2 xmm0, xmm0, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha256rnds2_all_xmm0 ENDP

; --- SHA256MSG1 ---

op_sha256msg1_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA256
    sha256msg2 xmm0, xmm1
ELSE
    sha256msg1 xmm0, xmm1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha256msg1_reg ENDP

op_sha256msg1_mem PROC FRAME
    .endprolog
    movdqa xmm4, xmmword ptr [rcx]
    sha256msg1 xmm4, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm4
    ret
op_sha256msg1_mem ENDP

op_sha256msg1_self PROC FRAME
    .endprolog
    movdqu xmm5, xmmword ptr [rcx]
    sha256msg1 xmm5, xmm5
    movdqu xmmword ptr [r8], xmm5
    ret
op_sha256msg1_self ENDP

; --- SHA256MSG2 (sequential lanes 2..3 depend on updated lanes 0..1) ---

op_sha256msg2_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA256
    sha256msg1 xmm0, xmm1
ELSE
    sha256msg2 xmm0, xmm1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha256msg2_reg ENDP

op_sha256msg2_mem PROC FRAME
    .endprolog
    movdqa xmm2, xmmword ptr [rcx]
    sha256msg2 xmm2, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm2
    ret
op_sha256msg2_mem ENDP

op_sha256msg2_self PROC FRAME
    .endprolog
    movdqu xmm4, xmmword ptr [rcx]
    sha256msg2 xmm4, xmm4
    movdqu xmmword ptr [r8], xmm4
    ret
op_sha256msg2_self ENDP

; --- Multi-instruction pipeline using XMM0..XMM5, [r8] reload, and chained SHA256RNDS2 ---

op_sha256_pipeline PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
    movdqa xmm2, xmm0
    sha256msg1 xmm2, xmm1
IFDEF ALTER_SHA256
    sha256msg1 xmm2, xmm1
ELSE
    sha256msg2 xmm2, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm2
    movdqu xmm3, xmm0
    movdqa xmm4, xmm1
    movdqa xmm0, xmmword ptr [r8]
    sha256rnds2 xmm3, xmm4, xmm0
    movdqa xmm5, xmm4
    sha256rnds2 xmm5, xmm3, xmm0
    movdqa xmmword ptr [r8], xmm5
    ret
op_sha256_pipeline ENDP

; --- Discarded faulting m128 read must not be dropped under /O2 ---

op_sha256_discarded_read PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
    movdqa xmm5, xmm1
    sha256msg1 xmm5, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm1
    ret
op_sha256_discarded_read ENDP

END
