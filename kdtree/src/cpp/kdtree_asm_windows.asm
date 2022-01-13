include <tournament_tree_asm_windows.asm>

.code

save_xmm_registers MACRO rfp: REQ, offset: REQ, regs :VARARG
    count = 0
    FOR reg,<regs>
        vmovaps [rfp + offset + 16 * count], reg
        .savexmm128 reg, offset + 16 * count
        count = count + 1
    ENDM
ENDM

restore_xmm_registers MACRO rfp: REQ, offset: REQ, regs :VARARG
    count = 0
    FOR reg,<regs>
        vmovaps reg, [rfp + offset + 16 * count]
        count = count + 1
    ENDM
ENDM


insert_closest_m MACRO load_query, compute_distance, compute_distance_single
    LOCAL loop_start, scalar_insert_start, scalar_insert_end, loop_end, tail_loop, tail_loop_start, tail_end

    load_query r8

    ; Load current best distance from tree
    vbroadcastss current_max_reg, DWORD PTR [r9]

    ; Load end pointer into rdx
    lea rdx, [rdx * 8]
    lea rdx, [rcx + rdx * 2 - 7 * 16]

    cmp rcx, rdx
    jae tail_loop

loop_start:
    compute_distance
    compare_distances ebx, current_max_reg, loop_end
scalar_insert_start:
    scalar_insert_test ebx, current_max_reg_xmm, scalar_insert_end

    ; update tournament tree with new best
    mov esi, [indices_buffer + r10]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_branchless_m r9, rdi, rsi, r11, r12, xmm3

    ; reload top value, xmm0 is updated by the tournament tree macros
    vbroadcastss current_max_reg, xmm0
scalar_insert_end:
    scalar_insert_loop ebx, scalar_insert_start
loop_end:
    lea rcx, [rcx + 128]
    cmp rcx, rdx
    jb loop_start


tail_loop:
; Adjust end pointer to non-truncated value
    add rdx, 7 * 16
; Test for any iterations in tail loop
    cmp rcx, rdx
    je done
; Tail loop (remainder)
    prepare_tail_loop current_max_reg_xmm
tail_loop_start:
    compute_distance_single xmm2
    ucomiss xmm0, current_max_reg_xmm
    jae tail_end

    mov esi, DWORD PTR[rcx + 12]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_branchless_m r9, rdi, rsi, r11, r12, xmm3

    vmovaps current_max_reg_xmm, xmm0
tail_end:
    add rcx, 16
    cmp rcx, rdx
    jb tail_loop_start
ENDM

save_all_prolog MACRO
    pad_size = 8
    stack_size = locals_size + pad_size + 10 * 16

    FOR reg, <rbp,rbx,rdi,rsi,r12,r13,r14,r15>
        push reg
        .pushreg reg
    ENDM
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
.setframe rsp, 0
    save_xmm_registers rsp, locals_size, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15
.endprolog
ENDM

restore_all_epilog MACRO
    restore_xmm_registers rsp, locals_size, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15
    vzeroupper
    add rsp, stack_size
    FOR reg, <r15,r14,r13,r12,rsi,rdi,rbx,rbp>
        pop reg
    ENDM
ENDM

insert_closest_avx2_m MACRO compute_distance
    stack_arguments_offset = stack_size + 8 * 8 + 8
    distances_buffer EQU rbp

    test rdx, rdx
    je done

    ; Align rbp to 32-byte boundary
    lea rbp, [rsp + 16]
    and rbp, -32

    ; Load current best distance
    vbroadcastss ymm8, DWORD PTR[r9]

    ; Load query vector
    vbroadcastss ymm5, DWORD PTR[r8]
    vbroadcastss ymm6, DWORD PTR[r8 + 4]
    vbroadcastss ymm7, DWORD PTR[r8 + 8]

    ; Load start pointers
    mov r12, [rcx]
    mov r13, [rcx + 8]
    mov r14, [rcx + 16]

    ; Load indices pointer into r15
    mov r15, [rsp + stack_arguments_offset + 4 * 8]

    ; Compute end pointer
    lea rdx, [r12 + rdx * 4]
loop_start:
    vmovaps ymm2, YMMWORD PTR [r12]
    vmovaps ymm3, YMMWORD PTR [r13]
    vmovaps ymm4, YMMWORD PTR [r14]

    compute_distance

    vcmpltps ymm3, ymm2, ymm8
    vmovmskps ebx, ymm3

    test bl, bl
    je loop_test

    vmovaps [distances_buffer], ymm2
    xor ecx, ecx
scalar_insert_start:
    test bl, 1
    je scalar_insert_end

    vmovss xmm0, DWORD PTR[distances_buffer + 4 * rcx]
    ucomiss xmm0, xmm8
    jae scalar_insert_end

    ; Perform scalar insert
    mov esi, DWORD PTR[r15 + 4 * rcx]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_branchless_m r9, rdi, rsi, r10, r11, xmm4
    vbroadcastss ymm8, xmm0
scalar_insert_end:
    inc ecx
    shr ebx, 1
    test bl, bl
    jne scalar_insert_start
loop_test:
    ; Reload array pointers
    add r12, 4 * 8
    add r13, 4 * 8
    add r14, 4 * 8
    add r15, 4 * 8
    cmp r12, rdx
    jb loop_start
done:
ENDM

compute_distance_l2_soa MACRO
    vsubps ymm2, ymm2, ymm5
    vsubps ymm3, ymm3, ymm6
    vsubps ymm4, ymm4, ymm7

    vmulps ymm2, ymm2, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm4, ymm4, ymm4

    vaddps ymm2, ymm2, ymm3
    vaddps ymm2, ymm2, ymm4
ENDM

; Main procedure for inserting closest points into given tournament tree.
;
; Arguments:
;   - rcx: address of array of points, index pairs
;   - rdx: number of points in array
;   - r8: address of query vector
;   - r9: address of the tournament tree
;   - (on stack): address of indices
wenda_insert_closest_l2_avx2 PROC PUBLIC FRAME
    locals_size = 32 + 16

    save_all_prolog
    insert_closest_avx2_m compute_distance_l2_soa
    restore_all_epilog
    ret
wenda_insert_closest_l2_avx2 ENDP


compute_distance_l2_periodic_soa MACRO
    vsubps ymm2, ymm2, ymm5
    vsubps ymm3, ymm3, ymm6
    vsubps ymm4, ymm4, ymm7

    FOR reg,<ymm2,ymm3,ymm4>
        vaddps ymm11, reg, ymm10
        vsubps ymm12, reg, ymm10
        vmulps reg, reg, reg
        vmulps ymm11, ymm11, ymm11
        vmulps ymm12, ymm12, ymm12
        vminps reg, reg, ymm11
        vminps reg, reg, ymm12
    ENDM

    vaddps ymm2, ymm2, ymm3
    vaddps ymm2, ymm2, ymm4
ENDM

wenda_insert_closest_l2_periodic_avx2 PROC PUBLIC FRAME
    locals_size = 32 + 16

    save_all_prolog

    distances_buffer EQU rbp
    stack_arguments_offset = stack_size + 8 * 8 + 8

    ; load periodic boundary from argument (on stack)
    vbroadcastss ymm10, DWORD PTR[rsp + stack_arguments_offset + 5 * 8]

    insert_closest_avx2_m compute_distance_l2_periodic_soa

    restore_all_epilog
    ret

wenda_insert_closest_l2_periodic_avx2 ENDP

ALIGN 16
query_mask DWORD -1, -1, -1, 0
flt_max DWORD 07f7fffffr

END
