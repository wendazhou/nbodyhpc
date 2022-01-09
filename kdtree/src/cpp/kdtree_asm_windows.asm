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
    mov DWORD PTR[ptr_tree + 4], r9d
    mov DWORD PTR[ptr_tree + 8], r10d
    ret
ENDM


; Procedure for replacing top element
; C prototype: void tournament_tree_update_root(tournament_tree_t *tree, uint32_t index, float element_value uint32_t element_idx)
; 
; Arguments are expected as follows:
;   rcx: address of tournament tree
;   edx: index at which element is placed
;   xmm0: value of the inserted element
;   r9d: index of the inserted element
tournament_tree_update_root PROC PUBLIC
    tournament_tree_update_root_m rcx, edx, r9d, r10d, r8d, r11d
tournament_tree_update_root ENDP
END
