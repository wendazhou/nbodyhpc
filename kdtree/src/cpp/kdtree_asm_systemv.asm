default rel

%macro tournament_tree_update_root_m 5
    %define reg_tree %1
    %define reg_idx %2
    %define reg_v %3

    ; Load initial index of element
    mov %4, reg_idx
    shl %4, 32
    or reg_v, %4

    cmp reg_idx, 1
    jbe %%finish
%%loop_start:
    shr reg_idx, 1

    ; Compute address at reg_trere + 12 * reg_idx
    lea rax, [reg_idx + 2 * reg_idx]
    lea rax, [reg_tree + 4 * rax]

    vmovss xmm1, DWORD [rax]
    ucomiss xmm1, xmm0

    ; Branchless assign xmm0 to be winner, xmm9 to be loser
    vminss xmm9, xmm0, xmm1
    vmaxss xmm0, xmm0, xmm1
    vmovaps xmm1, xmm9

    ; Store current winner as loser, load stored winner
    mov %4, QWORD[rax + 4]
    mov %5, %4
    cmova %5, reg_v
    mov QWORD [rax + 4], %5
    cmova reg_v, %4

    vmovss DWORD [rax], xmm1

    cmp reg_idx, 1
    ja %%loop_start
%%finish:
    vmovss DWORD [reg_tree], xmm0
    mov QWORD [reg_tree + 4], reg_v
%endmacro

; Macro implementation of swapping with top element in the tree.
;    This macro reads out the index of the top element from the top of the tree,
;    and then writes the given values (in fv_reg, sv_reg) to that element's location.
%macro tournament_tree_swap_top_m 5
    %define reg_tree %1
    %define reg_out_idx %2
    %define reg_fv %3
    %define reg_sv %4
    %define reg_tmp_addr %5

    mov reg_out_idx, DWORD [reg_tree + 8]
    lea reg_tmp_addr, [reg_out_idx + reg_out_idx * 2]
    lea reg_tmp_addr, [reg_tree + reg_tmp_addr * 4]
    vmovss DWORD[reg_tmp_addr], reg_fv
    mov DWORD[reg_tmp_addr + 4], reg_sv
%endmacro

global tournament_tree_update_root
tournament_tree_update_root:
    tournament_tree_update_root_m rdi, rsi, rdx, r8, r9
    vzeroupper
    ret

global tournament_tree_replace_top
tournament_tree_replace_top:
    mov edx, esi
    tournament_tree_swap_top_m rdi, esi, xmm0, edx, rax
    jmp tournament_tree_update_root ; tail call


%macro compute_distance_l2 0
    vsubps ymm2, ymm2, ymm5
    vsubps ymm3, ymm3, ymm6
    vsubps ymm4, ymm4, ymm7

    vmulps ymm2, ymm2, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm4, ymm4, ymm4

    vaddps ymm2, ymm2, ymm3
    vaddps ymm2, ymm2, ymm4
%endmacro

%macro compute_l2_periodic_squares_impl 1-*
    ; Computes the 1-d L2 distance with periodic boundary conditions for each of the registers
    ; This function expects each of the input registers to contain the difference value.
    ; This function expects ymm10 to hold the box size
    ; This function uses ymm1, ymm12 as scratch registers

    %rep %0

    vsubps ymm11, %1, ymm10
    vaddps ymm12, %1, ymm10
    vmulps %1, %1, %1
    vmulps ymm11, ymm11, ymm11
    vmulps ymm12, ymm12, ymm12
    vminps %1, %1, ymm11
    vminps %1, %1, ymm12

    %rotate 1

    %endrep
%endmacro

%macro compute_distance_l2_periodic 0
    vsubps ymm2, ymm2, ymm5
    vsubps ymm3, ymm3, ymm6
    vsubps ymm4, ymm4, ymm7

    compute_l2_periodic_squares_impl ymm2, ymm3, ymm4

    vaddps ymm2, ymm2, ymm3
    vaddps ymm2, ymm2, ymm4
%endmacro

%macro insert_closest_l2_avx2 2
    %define compute_distance %1
    %define done_label %2

    %define reg_query rdx
    %define reg_tree rcx
    %define reg_indices r8
    %define distances_buffer rsp

    test rsi, rsi
    je done_label

    ; Load current best distance
    vbroadcastss ymm8, DWORD [reg_tree]

    ; Load query vector
    vbroadcastss ymm5, DWORD [reg_query]
    vbroadcastss ymm6, DWORD [reg_query + 4]
    vbroadcastss ymm7, DWORD [reg_query + 8]

    ; Load start pointers
    mov r12, [rdi]
    mov r13, [rdi + 8]
    mov r14, [rdi + 16]

    ; Compute end pointer
    lea rsi, [r12 + rsi * 4]
%%loop_start:
    vmovaps ymm2, YWORD [r12]
    vmovaps ymm3, YWORD [r13]
    vmovaps ymm4, YWORD [r14]

    compute_distance

    vcmpltps ymm3, ymm2, ymm8
    vmovmskps ebx, ymm3

    test bl, bl
    je %%loop_test

    vmovaps [distances_buffer], ymm2
    xor rdi, rdi
%%scalar_insert_start:
    test bl, 1
    je %%scalar_insert_end

    vmovss xmm0, DWORD [distances_buffer + 4 * rdi]
    ucomiss xmm0, xmm8
    jae %%scalar_insert_end

    ; Perform scalar insert
    mov r15d, DWORD [reg_indices + 4 * rdi]
    tournament_tree_swap_top_m reg_tree, r9d, xmm0, r15d, r11
    tournament_tree_update_root_m reg_tree, r9, r15, r10, r11
    vbroadcastss ymm8, xmm0
%%scalar_insert_end:
    inc rdi
    shr ebx, 1
    test bl, bl
    jne %%scalar_insert_start
%%loop_test:
    ; Reload array pointers
    add r12, 4 * 8
    add r13, 4 * 8
    add r14, 4 * 8
    add reg_indices, 4 * 8
    cmp r12, rsi
    jb %%loop_start
%endmacro

global wenda_insert_closest_l2_avx2
wenda_insert_closest_l2_avx2:
    %assign stack_size 32

    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp
    mov rbp, rsp

    ; allocate stack and align to 32-byte boundary
    sub rsp, stack_size
    and rsp, -32

    insert_closest_l2_avx2 compute_distance_l2, .done
.done:
    vzeroupper
    leave
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx

    ret

global wenda_insert_closest_l2_periodic_avx2
wenda_insert_closest_l2_periodic_avx2:
    %assign stack_size 32

    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp
    mov rbp, rsp

    ; allocate stack and align to 32-byte boundary
    sub rsp, stack_size
    and rsp, -32

    ; Broadcasts the box size from its original register to ymm10
    vbroadcastss ymm10, xmm0

    insert_closest_l2_avx2 compute_distance_l2_periodic, .done
.done:
    vzeroupper
    leave
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx

    ret

align 16
query_mask:
    dd -1
    dd -1
    dd -1
    dd 0
flt_max:
    dd 07f7fffffh
