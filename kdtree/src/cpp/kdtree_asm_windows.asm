include <tournament_tree_asm_windows.asm>

.code


restore_xmm_registers MACRO offset: REQ, regs :VARARG
    count = 0
    FOR reg,<regs>
        vmovaps reg, [rbp + offset + 16 * count]
        count = count + 1
    ENDM
ENDM

; Macro for loading and computing 
load_and_compute_l2 MACRO query_reg
    vmovups ymm3, YMMWORD PTR [rcx]
    vmovups ymm4, YMMWORD PTR [rcx + 32]
    vmovups ymm5, YMMWORD PTR [rcx + 64]
    vmovups ymm6, YMMWORD PTR [rcx + 96]

    vpunpckhdq ymm7, ymm3, ymm5
    vpunpckhdq ymm8, ymm4, ymm6
    vpunpckhdq ymm7, ymm7, ymm8

    vsubps ymm3, ymm3, query_reg
    vsubps ymm4, ymm4, query_reg
    vsubps ymm5, ymm5, query_reg
    vsubps ymm6, ymm6, query_reg

    vdpps ymm3, ymm3, ymm3, 01110001b
    vdpps ymm4, ymm4, ymm4, 01110010b
    vdpps ymm5, ymm5, ymm5, 01110100b
    vdpps ymm6, ymm6, ymm6, 01111000b

    vaddps ymm3, ymm3, ymm4
    vaddps ymm5, ymm5, ymm6
    vaddps ymm3, ymm3, ymm5
ENDM

; Loads query vector and duplicates across lanes
;   - reg_source: register containing the address of the query vector to load
;   - reg_idx: integer representing the xmm register to load into
;   - query_mask: address of an aligned constant mask DWORD -1, -1, -1, 0
load_query_vector MACRO reg_source, reg_idx, query_mask
    query_reg_xmm CATSTR <xmm>, %reg_idx
    query_reg_ymm CATSTR <ymm>, %reg_idx

    vmovdqa query_reg_xmm, XMMWORD PTR [query_mask]
    vmaskmovps query_reg_xmm, query_reg_xmm, XMMWORD PTR [reg_source]
    vinsertf128 query_reg_ymm, query_reg_ymm, query_reg_xmm, 1
ENDM


; Find index of element closest to given query in array
wenda_find_closest_l2_avx2 PROC PUBLIC FRAME
    ; Stack-based variables
    stack_size = 64 + 4 * 16

    push rbp
.pushreg rbp
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
    mov rbp, rsp
.setframe rbp, 0
    save_xmm_registers 64, xmm6, xmm7, xmm8, xmm9
.endprolog

    indices_buffer EQU rbp
    distances_buffer EQU rbp + 32

    ; Load query vector into xmm0, and duplicate across lanes
    load_query_vector r8, 2, query_mask

    ; Load current best distance
    vbroadcastss ymm9, DWORD PTR [flt_max]

    ; Load end pointer into rdx
    lea rdx, [rdx * 8]
    lea rdx, [rcx + rdx * 2 - 7 * 16]

loop_start:
    load_and_compute_l2 ymm2

    vmovups YMMWORD PTR [indices_buffer], ymm7

    vcmpltps ymm7, ymm3, ymm9
    vmovmskps eax, ymm7
    test eax, eax
    je loop_end

    vmovups YMMWORD PTR [distances_buffer], ymm3
    xor r10, r10
scalar_insert_start:
    test eax, 1
    je scalar_insert_end

    ; perform scalar insertion
    vmovss xmm0, DWORD PTR [distances_buffer + 4 * r10]
    ucomiss xmm0, xmm9
    jae scalar_insert_end

    vbroadcastss ymm9, xmm0
    mov r11d, [indices_buffer + 4 * r10]
scalar_insert_end:
    shr eax, 1
    inc r10
    test eax, eax
    jne scalar_insert_start
loop_end:
    lea rcx, [rcx + 128]
    cmp rcx, rdx
    jb loop_start

; Store best result so far into eax
    mov eax, r11d

; Adjust end pointer to non-truncated value
    add rdx, 7 * 16
; Test for tail loop
    cmp rcx, rdx
    je done

; Tail loop (remainder)
    vmovaps xmm5, xmm9
tail_start:
    vmovups xmm3, XMMWORD PTR [rcx]
    vsubps xmm3, xmm3, xmm2
    vdpps xmm3, xmm3, xmm3, 01110001b
    ucomiss xmm3, xmm5
    jae tail_end

    vbroadcastss xmm5, xmm3
    mov eax, DWORD PTR[rcx + 12]
tail_end:
    add rcx, 16
    cmp rcx, rdx
    jb tail_start

done:
; Epilog
    restore_xmm_registers 64, xmm6, xmm7, xmm8, xmm9
    add rsp, stack_size
    pop rbp

    ret

wenda_find_closest_l2_avx2 ENDP

; Main procedure for finding k closest points to query vector,
;
; Arguments:
;   - rcx: address of array of points, index pairs
;   - rdx: number of points in array
;   - r8: address of query vector
;   - r9: address of the tournament tree
wenda_insert_closest_l2_avx2 PROC PUBLIC FRAME
    stack_size = 64 + 4 * 16

    FOR reg, <rbp,rbx,rdi,rsi,r12,r13,r14>
        push reg
        .pushreg reg
    ENDM
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
    mov rbp, rsp
.setframe rbp, 0
    save_xmm_registers 64, xmm6, xmm7, xmm8, xmm9
.endprolog

    indices_buffer EQU rbp
    distances_buffer EQU rbp + 32

    load_query_vector r8, 2, query_mask

    ; Load current best distance from tree
    vbroadcastss ymm9, DWORD PTR [r9]

    ; Load end pointer into rdx
    lea rdx, [rdx * 8]
    lea rdx, [rcx + rdx * 2 - 7 * 16]

loop_start:
    load_and_compute_l2 ymm2

    vmovups YMMWORD PTR [indices_buffer], ymm7

    vcmpltps ymm7, ymm3, ymm9
    vmovmskps ebx, ymm7
    test ebx, ebx
    je loop_end

    vmovups YMMWORD PTR [distances_buffer], ymm3
    xor r10, r10

scalar_insert_start:
    test ebx, 1
    je scalar_insert_end

    ; test for current best
    vmovss xmm0, DWORD PTR [distances_buffer + r10]
    ucomiss xmm0, xmm9
    jae scalar_insert_end

    ; update tournament tree with new best
    mov esi, [indices_buffer + r10]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_m r9, edi, esi, r11d, r12d, r13d

    ; reload top value, xmm0 is updated by the tournament tree macros
    vbroadcastss ymm9, xmm0
scalar_insert_end:
    shr ebx, 1
    add r10, 4
    test ebx, ebx
    jne scalar_insert_start

loop_end:
    lea rcx, [rcx + 128]
    cmp rcx, rdx
    jb loop_start

; epilog
    restore_xmm_registers 64, xmm6, xmm7, xmm8, xmm9
    add rsp, stack_size
    FOR reg, <r14,r13,r12,rsi,rdi,rbx,rbp>
        pop reg
    ENDM
    ret
wenda_insert_closest_l2_avx2 ENDP

ALIGN 16
query_mask DWORD -1, -1, -1, 0
flt_max DWORD 07f7fffffr

END
