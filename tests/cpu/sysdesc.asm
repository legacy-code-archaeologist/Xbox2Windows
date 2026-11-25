; coverage: source cpu_aot_sysdesc
; Source-built MASM fixture for opt-in --host-cpu system-descriptor and machine-status
; store instructions: SGDT, SIDT, SLDT, SMSW, and STR (Windows x64 leaf ABI:
; RCX, RDX, R8 -> RAX).
.code

; --- SGDT / SIDT (10-byte m16&64 descriptor stores) ---
; ALTER_IMM swaps SGDT <-> SIDT so mutant divergence is directly observed.
op_sgdt PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    sidt FWORD PTR [r8]
ELSE
    sgdt FWORD PTR [r8]
ENDIF
    mov rax, rdx
    ret
op_sgdt ENDP

op_sgdt_disp PROC FRAME
    .endprolog
    sgdt FWORD PTR [r8+16]
    mov rax, rdx
    ret
op_sgdt_disp ENDP

op_sidt PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    sgdt FWORD PTR [r8]
ELSE
    sidt FWORD PTR [r8]
ENDIF
    mov rax, rdx
    ret
op_sidt ENDP

op_sidt_disp PROC FRAME
    .endprolog
    sidt FWORD PTR [r8+16]
    mov rax, rdx
    ret
op_sidt_disp ENDP

; --- SLDT (r16, r32, r64, m16) ---
op_sldt_r16 PROC FRAME
    .endprolog
    mov rax, rdx
IFDEF ALTER_IMM
    str ax
ELSE
    sldt ax
ENDIF
    ret
op_sldt_r16 ENDP

op_sldt_r32 PROC FRAME
    .endprolog
    mov rax, rdx
    sldt eax
    ret
op_sldt_r32 ENDP

op_sldt_r64 PROC FRAME
    .endprolog
    mov rax, rdx
    sldt rax
    ret
op_sldt_r64 ENDP

op_sldt_m16 PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    str WORD PTR [r8]
ELSE
    sldt WORD PTR [r8]
ENDIF
    mov rax, rdx
    ret
op_sldt_m16 ENDP

op_sldt_m16_disp PROC FRAME
    .endprolog
    sldt WORD PTR [r8+4]
    mov rax, rdx
    ret
op_sldt_m16_disp ENDP

; --- STR (r16, r32, r64, m16) ---
op_str_r16 PROC FRAME
    .endprolog
    mov rax, rdx
IFDEF ALTER_IMM
    sldt ax
ELSE
    str ax
ENDIF
    ret
op_str_r16 ENDP

op_str_r32 PROC FRAME
    .endprolog
    mov rax, rdx
    str eax
    ret
op_str_r32 ENDP

op_str_r64 PROC FRAME
    .endprolog
    mov rax, rdx
    str rax
    ret
op_str_r64 ENDP

op_str_m16 PROC FRAME
    .endprolog
    str WORD PTR [r8]
    mov rax, rdx
    ret
op_str_m16 ENDP

op_str_m16_disp PROC FRAME
    .endprolog
    str WORD PTR [r8+6]
    mov rax, rdx
    ret
op_str_m16_disp ENDP

; --- SMSW (r16, r32, r64, m16) ---
op_smsw_r16 PROC FRAME
    .endprolog
    mov rax, rdx
IFDEF ALTER_IMM
    str ax
ELSE
    smsw ax
ENDIF
    ret
op_smsw_r16 ENDP

op_smsw_r32 PROC FRAME
    .endprolog
    mov rax, rdx
    smsw eax
    ret
op_smsw_r32 ENDP

op_smsw_r64 PROC FRAME
    .endprolog
    mov rax, rdx
    smsw rax
    ret
op_smsw_r64 ENDP

op_smsw_m16 PROC FRAME
    .endprolog
    smsw WORD PTR [r8]
    mov rax, rdx
    ret
op_smsw_m16 ENDP

op_smsw_m16_disp PROC FRAME
    .endprolog
    smsw WORD PTR [r8+8]
    mov rax, rdx
    ret
op_smsw_m16_disp ENDP

; --- Flag preservation across all 5 sysdesc instructions ---
op_sysdesc_flags_preserve PROC FRAME
    .endprolog
    mov r9d, 080000001h
    add r9d, 080000002h     ; r9d = 3 -> CF=1, OF=1, SF=0, ZF=0, PF=1
    mov r10, rdx
    sgdt FWORD PTR [r8+16]
    sidt FWORD PTR [r8+32]
    sldt r10w
    str r10w
    smsw r11d
    sete BYTE PTR [r8]
IFDEF ALTER_IMM
    setae BYTE PTR [r8+1]
ELSE
    setb BYTE PTR [r8+1]
ENDIF
    sets BYTE PTR [r8+2]
    seto BYTE PTR [r8+3]
    setp BYTE PTR [r8+4]
    mov rax, r11
    ret
op_sysdesc_flags_preserve ENDP

; --- Volatile store ordering under /O2 (NULL r8 must fault) ---
op_sysdesc_discarded_stores PROC FRAME
    .endprolog
    sgdt FWORD PTR [r8]
    sidt FWORD PTR [r8+16]
    sldt WORD PTR [r8+32]
    str WORD PTR [r8+34]
    smsw WORD PTR [r8+36]
    mov rax, rdx
    ret
op_sysdesc_discarded_stores ENDP

; --- Unused register writes compile cleanly under /O2 /W4 /WX ---
op_sysdesc_unused_reg PROC FRAME
    .endprolog
    sldt r9d
    str r10d
    smsw r11d
    mov rax, rdx
    ret
op_sysdesc_unused_reg ENDP

END
