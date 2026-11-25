; coverage: record cpu_aot_bmi1_bit
; Source-built MASM fixture for the three BMI1 bit-isolation/mask/reset forms from
; docs/coverage/x64_zen2.tsv group BMI/ADX: BLSI, BLSMSK, and BLSR in 32-bit and
; 64-bit register and flat [rcx|rdx|r8] memory forms, plus CF/ZF/SF/OF flag-packing,
; 32-bit zero-extension, [r8] source-then-store, and discarded faulting m64 read.
.code

PACK_CF_ZF_SF_OF MACRO
    setb r9b
    sete r10b
    sets r11b
    seto byte ptr [r8]
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    movzx r11d, r11b
    shl r11d, 2
    or r9d, r11d
    movzx r10d, byte ptr [r8]
    shl r10d, 3
    or r9d, r10d
    mov dword ptr [r8], r9d
ENDM

; --- Register forms (32-bit & 64-bit; mov rax,-1 before 32-bit tests [63:32] zero-ext) ---

op_blsi32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsr eax, ecx
ELSE
    blsi eax, ecx
ENDIF
    ret
op_blsi32 ENDP

op_blsi64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsr rax, rcx
ELSE
    blsi rax, rcx
ENDIF
    ret
op_blsi64 ENDP

op_blsmsk32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsi eax, ecx
ELSE
    blsmsk eax, ecx
ENDIF
    ret
op_blsmsk32 ENDP

op_blsmsk64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsi rax, rcx
ELSE
    blsmsk rax, rcx
ENDIF
    ret
op_blsmsk64 ENDP

op_blsr32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsmsk eax, ecx
ELSE
    blsr eax, ecx
ENDIF
    ret
op_blsr32 ENDP

op_blsr64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsmsk rax, rcx
ELSE
    blsr rax, rcx
ENDIF
    ret
op_blsr64 ENDP

; --- Flat [rcx|rdx] m32 / m64 source forms ---

op_blsi32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsr eax, dword ptr [rcx]
ELSE
    blsi eax, dword ptr [rcx]
ENDIF
    ret
op_blsi32_mem ENDP

op_blsi64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsr rax, qword ptr [rdx]
ELSE
    blsi rax, qword ptr [rdx]
ENDIF
    ret
op_blsi64_mem ENDP

op_blsmsk32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsi eax, dword ptr [rdx]
ELSE
    blsmsk eax, dword ptr [rdx]
ENDIF
    ret
op_blsmsk32_mem ENDP

op_blsmsk64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsi rax, qword ptr [rcx]
ELSE
    blsmsk rax, qword ptr [rcx]
ENDIF
    ret
op_blsmsk64_mem ENDP

op_blsr32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1
    blsmsk eax, dword ptr [rcx]
ELSE
    blsr eax, dword ptr [rcx]
ENDIF
    ret
op_blsr32_mem ENDP

op_blsr64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1
    blsmsk rax, qword ptr [rdx]
ELSE
    blsr rax, qword ptr [rdx]
ENDIF
    ret
op_blsr64_mem ENDP

; --- Flag-packing exports: (CF << 0) | (ZF << 1) | (SF << 2) | (OF << 3) stored at [r8] ---

op_blsi32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    blsi eax, ecx
    PACK_CF_ZF_SF_OF
    ret
op_blsi32_flags ENDP

op_blsi64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    blsi rax, rcx
    PACK_CF_ZF_SF_OF
    ret
op_blsi64_flags ENDP

op_blsmsk32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    blsmsk eax, ecx
    PACK_CF_ZF_SF_OF
    ret
op_blsmsk32_flags ENDP

op_blsmsk64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    blsmsk rax, rcx
    PACK_CF_ZF_SF_OF
    ret
op_blsmsk64_flags ENDP

op_blsr32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    blsr eax, ecx
    PACK_CF_ZF_SF_OF
    ret
op_blsr32_flags ENDP

op_blsr64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    blsr rax, rcx
    PACK_CF_ZF_SF_OF
    ret
op_blsr64_flags ENDP

op_blsr64_mem_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    blsr rax, qword ptr [rcx]
    PACK_CF_ZF_SF_OF
    ret
op_blsr64_mem_flags ENDP

; --- [r8] source-then-store and discarded faulting m64 read ---

op_r8_source_then_store PROC FRAME
    .endprolog
    blsi r9, qword ptr [r8]
    blsmsk r10, qword ptr [rcx]
    blsr r11, qword ptr [rdx]
    or r9, r10
    xor r9, r11
    mov qword ptr [r8], r9
    mov rax, r9
    ret
op_r8_source_then_store ENDP

op_bmi1_discarded_read PROC FRAME
    .endprolog
    blsi r9, qword ptr [rdx]
    mov rax, rcx
    ret
op_bmi1_discarded_read ENDP

END
