.code

; Implementation of updating the path to the root element in tournament tree.
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

; Implementation of updating the path to the root element in tournament tree.
;   Specialized for a tournament tree of float, uint32_t pairs with comparison
;   only induced by the first element. This implementation is uses a single 64-bit operand.
;   Note: due to the size of the records, this will cause an unaligned access about half
;       of the time. Better packing for the records could improve performance here.
;
; For flexibility, this is a macro, which requires the following
; arguments:
;   - ptr_tree: MACHINE register containing the address of the tree
;   - reg_index: QWORD register containing the index of the element to replace
;   - reg_element_index: QWORD register containing the element index value
;   - reg_tmp1: (temporary) QWORD register
;
; Arguments marked temporary do not need to be set to a particular value.
; All arguments are clobbered by the macro, except for ptr_tree which retains its original value.
; Additionally, rax, xmm0 and xmm1 are clobbered by the macro.
;
tournament_tree_update_root_fused_m MACRO ptr_tree, reg_index, reg_element_idx, reg_tmp1
    LOCAL loop_start, loop_check, finish
    cmp reg_index, 1
    jbe finish

    ; Load initial index of element into ecx
    mov reg_tmp1, reg_index
    shl reg_tmp1, 32
    or reg_element_idx, reg_tmp1

loop_start:
    shr reg_index, 1

    ; Compute address at ptr_tree + 12 * reg_index
    lea rax, [reg_index + 2 * reg_index]
    lea rax, [ptr_tree + 4 * rax]

    movss xmm1, DWORD PTR [rax]
    ucomiss xmm1, xmm0
    jbe loop_check

    ; Store current winner as loser, load stored winner
    mov reg_tmp1, QWORD PTR[rax + 4]
    mov QWORD PTR [rax + 4], reg_element_idx
    mov reg_element_idx, reg_tmp1

    movss DWORD PTR [rax], xmm0
    movss xmm0, xmm1

loop_check:
    cmp reg_index, 1
    ja loop_start
finish:
    movss DWORD PTR[ptr_tree], xmm0
    mov QWORD PTR[ptr_tree + 4], reg_element_idx
ENDM

; Macro for swapping top element in the tree
; Arguments:
;  - tt_reg: register containing the address of the tree
;  - out_idx_reg: (output) register which will contain the index of the top element
;  - fv_reg: xmm register containing the first value of the element
;  - sv_reg: register containing the second value of the element
;  - out_address_reg: temporary register, will contain the address of the top element
tournament_tree_swap_top_m MACRO tt_reg, out_idx_reg, fv_reg, sv_reg, out_address_reg
    mov out_idx_reg, DWORD PTR [tt_reg + 8]
    lea out_address_reg, [out_idx_reg + out_idx_reg * 2]
    lea out_address_reg, [tt_reg + out_address_reg * 4]
    vmovss DWORD PTR[out_address_reg], fv_reg
    mov DWORD PTR[out_address_reg + 4], sv_reg
ENDM

save_xmm_registers MACRO offset: REQ, regs :VARARG
    count = 0
    FOR reg,<regs>
        vmovaps [rbp + offset + 16 * count], reg
        .savexmm128 reg, offset + 16 * count
        count = count + 1
    ENDM
ENDM

; Procedure for updating root in tournament tree
; C prototype: void tournament_tree_update_root(tournament_tree_t *tree, uint32_t index, float element_value uint32_t element_idx)
; 
; Arguments are expected as follows:
;   rcx: address of tournament tree
;   edx: index at which element is placed
;   xmm0: first value of the inserted element
;   r9d: second value of the inserted element
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
    tournament_tree_swap_top_m rcx, edx, xmm0, r8d, rax
    mov r9d, r8d
    jmp tournament_tree_update_root ; tail call
tournament_tree_replace_top ENDP
