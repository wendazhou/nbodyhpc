.code

; Implementation of replacing top element in tournament tree
;   Specialized for a tournament tree of float, uint32_t pairs
;   with comparison only induced by the first element.
;
; For flexibility, this is a macro, which requires the following
; arguments:
;   - ptr_tree: MACHINE register containing the address of the tree
;   - reg_index: DWORD register containing the index of the element to replace
;   - reg_element_index: DWORD register containing the element index value
;   - reg_tmp_winner_idx: (temporary) DWORD register containing the index of the winner
;   - reg_tmp1: (temporary) DWORD register
;   - reg_tmp2: (temporary) DWORD register
;
; Arguments marked temporary do not need to be set to a particular value.
; All arguments are clobbered by the macro, except for ptr_tree which retains its original value.
; Additionally, rax, xmm0 and xmm1 are clobbered by the macro.
;
tournament_tree_update_root_m MACRO ptr_tree, reg_index, reg_element_idx, reg_tmp_winner_idx, reg_tmp1, reg_tmp2
    LOCAL loop_start, loop_check, finish
    cmp reg_index, 1
    jbe finish

    ; Load initial index of element into ecx
    mov reg_tmp_winner_idx, reg_index

    ; In the main loop, we maintain a copy of the current value,
    ; which is comprised of a triplet (float, uint32_t, uint32_t),
    ; with the first two values corresponding to the value stored
    ; in the tournament tree, and the last value corresponding to
    ; the original index of the element in the tournament tree.
    ; They are kept in registers xmm0, reg_element_idx, reg_tmp_winner_idx
loop_start:
    shr reg_index, 1

    ; Compute address at ptr_tree + 12 * reg_index
    lea rax, [reg_index + 2 * reg_index]
    lea rax, [ptr_tree + 4 * rax]

    movss xmm1, DWORD PTR [rax]
    ucomiss xmm1, xmm0
    jbe loop_check

    ; Store current winner as loser, load stored winner
    mov reg_tmp1, DWORD PTR[rax + 4]
    mov DWORD PTR [rax + 4], reg_element_idx
    mov reg_tmp2, DWORD PTR[rax + 8]
    mov DWORD PTR [rax + 8], reg_tmp_winner_idx
    mov reg_element_idx, reg_tmp1
    mov reg_tmp_winner_idx, reg_tmp2

    movss DWORD PTR [rax], xmm0
    movss xmm0, xmm1

loop_check:
    cmp reg_index, 1
    ja loop_start
finish:
    movss DWORD PTR[ptr_tree], xmm0
    mov DWORD PTR[ptr_tree + 4], reg_element_idx
    mov DWORD PTR[ptr_tree + 8], reg_tmp_winner_idx
ENDM

save_xmm_registers MACRO offset: REQ, regs :VARARG
    count = 0
    FOR reg,<regs>
        vmovaps [rbp + offset + 16 * count], reg
        .savexmm128 reg, offset + 16 * count
        count = count + 1
    ENDM
ENDM

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

    stack_size_find_closest EQU 64 + 4 * 16

    push rbp
.pushreg rbp
    sub rsp, stack_size_find_closest ; reserve space for local variables and non-volatile registers
.allocstack stack_size_find_closest
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
    add rsp, stack_size_find_closest
    pop rbp

    ret

wenda_find_closest_l2_avx2 ENDP

; Main procedure for finding k closest points to query vector,
;
; Arguments:
;   - rcx: address of array of points, index pairs
;   - rdx: number of points in array
;   - r8: address of query vector
;   - r9: 
wenda_insert_closest_l2_avx2 PROC PUBLIC FRAME
    stack_size EQU 64 + 3 * 16

    push rbp
.pushreg rbp
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
    mov rbp, rsp
.setframe rbp, 0
    save_xmm_registers 64, xmm6, xmm7, xmm8
.endprolog

    indices_buffer EQU rbp
    distances_buffer EQU rbp + 32

    load_query_vector r8, 2, query_mask
wenda_insert_closest_l2_avx2 ENDP

ALIGN 16
query_mask DWORD -1, -1, -1, 0
flt_max DWORD 07f7fffffr

; Procedure for updating root in tournament tree
; C prototype: void tournament_tree_update_root(tournament_tree_t *tree, uint32_t index, float element_value uint32_t element_idx)
; 
; Arguments are expected as follows:
;   rcx: address of tournament tree
;   edx: index at which element is placed
;   xmm0: value of the inserted element
;   r9d: index of the inserted element
tournament_tree_update_root PROC PUBLIC
    tournament_tree_update_root_m rcx, edx, r9d, r10d, r8d, r11d
    ret
tournament_tree_update_root ENDP

; Procedure for replacing top element in tournament tree
; C prototype: void tournament_tree_replace_top(tournament_tree_t *tree, float element_value uint32_t element_idx)
; Argumens are expected as follows:
;   rcx: address of tournament tree
;   xmm0: first value of the inserted element
;   r8d: second value of the inserted element
tournament_tree_replace_top PROC PUBLIC
    mov edx, DWORD PTR [rcx + 8]
    lea eax, [edx + 2 * edx]
    vmovss DWORD PTR[rcx + 4 * rax], xmm0
    mov DWORD PTR[rcx + 4 * rax + 4], r8d
    mov r9d, r8d
    jmp tournament_tree_update_root ; tail call
tournament_tree_replace_top ENDP
END
