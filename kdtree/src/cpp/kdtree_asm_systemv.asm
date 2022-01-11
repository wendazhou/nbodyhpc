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

    ; Compute address at rdi + 12 * rsi
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


%macro transpose_registers 0
    ; Transpose loaded values
    vshufps ymm8, ymm3, ymm4, 0x044
    vshufps ymm9, ymm3, ymm4, 0x0ee
    vshufps ymm10, ymm5, ymm6, 0x044
    vshufps ymm11, ymm5, ymm6, 0x0ee

    vshufps ymm3, ymm8, ymm10, 0x088
    vshufps ymm4, ymm8, ymm10, 0x0dd
    vshufps ymm5, ymm9, ymm11, 0x088
    vshufps ymm7, ymm9, ymm11, 0x0dd
%endmacro

%macro compute_l2 0-1 ymm3
    transpose_registers

    ; Compute differences
    vsubps ymm3, ymm3, ymm12
    vsubps ymm4, ymm4, ymm13
    vsubps ymm5, ymm5, ymm14

    ; Compute squares
    vmulps ymm3, ymm3, ymm3
    vmulps ymm4, ymm4, ymm4
    vmulps ymm5, ymm5, ymm5

    ; Compute sum
    vaddps ymm3, ymm3, ymm4
    vaddps %1, ymm3, ymm5
%endmacro

%macro compute_l2_periodic_squares_impl 1-*
    %rep %0

    vsubps ymm8, %1, ymm2
    vaddps ymm9, %1, ymm2
    vmulps %1, %1, %1
    vmulps ymm8, ymm8, ymm8
    vmulps ymm9, ymm9, ymm9
    vminps %1, %1, ymm8
    vminps %1, %1, ymm9

    %rotate 1

    %endrep
%endmacro

%macro compute_l2_periodic 0-1 ymm3
    transpose_registers

    ; Compute differences
    vsubps ymm3, ymm3, ymm12
    vsubps ymm4, ymm4, ymm13
    vsubps ymm5, ymm5, ymm14

    ; Compute squares (including periodic),
    ; and select smallest value
    compute_l2_periodic_squares_impl ymm3, ymm4, ymm5

    ; Compute sum
    vaddps ymm3, ymm3, ymm4
    vaddps %1, ymm3, ymm5
%endmacro

%macro compute_l2_single 0-1 xmm0
    vmovups %1, OWORD [rdi]
    vsubps %1, %1, xmm2
    vdpps %1, %1, %1, 01110001b
%endmacro

%macro compute_l2_periodic_single 0
    vmaskmovps xmm0, xmm8, OWORD [rdi]
    vsubps xmm0, xmm0, xmm2
    vsubps xmm3, xmm0, xmm6
    vaddps xmm4, xmm0, xmm6

    vmulps xmm0, xmm0, xmm0
    vmulps xmm3, xmm3, xmm3
    vmulps xmm4, xmm4, xmm4

    vminps xmm0, xmm0, xmm3
    vminps xmm0, xmm0, xmm4

    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
%endmacro

%macro load_query_vector 0
    vbroadcastss ymm12, DWORD [rdx]
    vbroadcastss ymm13, DWORD [rdx + 4]
    vbroadcastss ymm14, DWORD [rdx + 8]
%endmacro

%macro prepare_tail_loop 0
    vmovaps xmm5, xmm15
    vunpckhps xmm2, xmm12, xmm13
    vxorps xmm13, xmm13, xmm13
    vblendps xmm2, xmm2, xmm14, 0100b
    vblendps xmm2, xmm2, xmm13, 1000b
%endmacro

%macro prepare_tail_loop_periodic 0
    vmovaps xmm6, xmm2
    vmovaps xmm8, OWORD [rel query_mask]
    prepare_tail_loop
%endmacro

%macro find_closest_m 4
    %define compute_distance %1
    %define compute_distance_single %2
    %define prepare_tail %3
    %define epilog_label %4

    load_query_vector
    vbroadcastss ymm15, DWORD [rel flt_max]

    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

    ; Check if main loop required
    cmp rdi, rsi
    jae %%tail
%%loop_start:
    vmovups ymm3, YWORD [rdi]
    vmovups ymm4, YWORD [rdi + 32]
    vmovups ymm5, YWORD [rdi + 64]
    vmovups ymm6, YWORD [rdi + 96]

    compute_distance

    vcmpltps ymm8, ymm3, ymm15
    vmovmskps eax, ymm8
    test eax, eax
    je %%loop_end

    vmovdqa YWORD [rsp + 32], ymm7
    vmovaps YWORD [rsp], ymm3
    xor r10, r10
%%scalar_insert_start:
    test eax, 1
    je %%scalar_insert_end

    vmovss xmm0, DWORD [rsp + 4 * r10]
    ucomiss xmm0, xmm15
    jae %%scalar_insert_end

    vbroadcastss ymm15, xmm0
    mov r11d, [rsp + 32 + 4 * r10]
%%scalar_insert_end:
    shr eax, 1
    inc r10
    test eax, eax
    jne %%scalar_insert_start
%%loop_end:
    lea rdi, [rdi + 128]
    cmp rdi, rsi
    jb %%loop_start

    mov eax, r11d

%%tail: ; Tail handling start
    add rsi, 7 * 16 ; Adjust end pointer to non-truncated value
    cmp rdi, rsi ; Test if tail loop required
    je epilog_label ; epilog

    prepare_tail
%%tail_start:
    compute_distance_single
    ucomiss xmm0, xmm5
    jae %%tail_end

    vmovaps xmm5, xmm0
    mov eax, DWORD[rdi + 12]
%%tail_end:
    add rdi, 16
    cmp rdi, rsi
    jb %%tail_start
%endmacro

global wenda_find_closest_l2_avx2
wenda_find_closest_l2_avx2:
    push rbp
    mov rbp, rsp
    sub rsp, 64
    and rsp, -32

    find_closest_m compute_l2, compute_l2_single, prepare_tail_loop, .done
.done:
    vzeroupper
    leave
    ret

global wenda_find_closest_l2_periodic_avx2
wenda_find_closest_l2_periodic_avx2:
    push rbp
    mov rbp, rsp
    sub rsp, 64
    and rsp, -32

    vbroadcastss ymm2, xmm0
    find_closest_m compute_l2_periodic, compute_l2_periodic_single, prepare_tail_loop_periodic, .done
.done:
    vzeroupper
    leave
    ret


%macro insert_closest_m 4
    %define compute_distance %1
    %define compute_distance_single %2
    %define prepare_tail %3
    %define epilog_label %4
    load_query_vector

    ; Load current best distance from tree
    vbroadcastss ymm15, DWORD[rcx]

    ; Adjust rsi to point to end (adjusted for unroll size)
    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

    ; Check if main loop required
    cmp rdi, rsi
    jae %%tail

    ; Load first iteration data
    vmovups ymm3, YWORD [rdi]
    vmovups ymm4, YWORD [rdi + 32]
    vmovups ymm5, YWORD [rdi + 64]
    vmovups ymm6, YWORD [rdi + 96]
%%loop_start:
    ; Compute l2 distance for each pair, store results in ymm0. Indices are also extracted and stored in ymm7.
    compute_distance ymm0

    vcmpltps ymm8, ymm0, ymm15 ; Compute pointwise comparison of distances to current best
    vmovmskps ebx, ymm8 ; Store comparison result in ebx

    lea rdi, [rdi + 128] ; Compute pointer for next iteration
    cmp rdi, rsi
    jae %%test_scalar_required ; Skip loading next iteration

    ; Load next iteration data
    vmovups ymm3, YWORD [rdi]
    vmovups ymm4, YWORD [rdi + 32]
    vmovups ymm5, YWORD [rdi + 64]
    vmovups ymm6, YWORD [rdi + 96]
%%test_scalar_required:
    test ebx, ebx
    je %%loop_end ; skip scalar updates if all elements are worse than current best

    vmovaps YWORD [rsp + distances_buffer_offset], ymm0 ; Save computed distances to stack
    vmovdqa YWORD [rsp + indices_buffer_offset], ymm7 ; Save extracted indices to stack
    xor r10, r10
%%scalar_insert_start:  ; Start scalar loop
    test ebx, 1
    je %%scalar_insert_end

    ; Load distance from stack
    vmovss xmm0, DWORD [rsp + distances_buffer_offset + r10]
    ucomiss xmm0, xmm15
    jae %%scalar_insert_end

    mov r11d, DWORD [rsp + indices_buffer_offset + r10]
    tournament_tree_swap_top_m rcx, r12d, xmm0, r11d, rax
    tournament_tree_update_root_m rcx, r12, r11, r8, r9

    vbroadcastss ymm15, xmm0
%%scalar_insert_end:  ; Test scalar loop
    shr ebx, 1
    add r10, 4
    test ebx, ebx
    jne %%scalar_insert_start
%%loop_end:  ; Test main vectorized loop
    cmp rdi, rsi
    jb %%loop_start

%%tail:  ; Tail handling start
    add rsi, 7 * 16 ; Adjust rsi to point to actual end
    ; Check if tail handling required
    cmp rdi, rsi
    je epilog_label

    prepare_tail
%%tail_start:  ; Tail loop
    compute_distance_single
    ucomiss xmm0, xmm5
    jae %%tail_end

    mov ebx, DWORD [rdi + 12]
    tournament_tree_swap_top_m rcx, r12d, xmm0, ebx, r11
    tournament_tree_update_root_m rcx, r12, rbx, r8, r9

    vmovaps xmm5, xmm0
%%tail_end:  ; Tail loop check
    add rdi, 16
    cmp rdi, rsi
    jb %%tail_start
%endmacro

global wenda_insert_closest_l2_avx2
wenda_insert_closest_l2_avx2:
    %assign indices_buffer_offset 0
    %assign distances_buffer_offset 32
    %assign stack_size 64

    push rbx
    push r12
    push rbp
    mov rbp, rsp

    ; allocate stack and align to 32-byte boundary
    sub rsp, stack_size
    and rsp, -32

    insert_closest_m compute_l2, compute_l2_single, prepare_tail_loop, .done
.done:
    vzeroupper
    leave
    pop r12
    pop rbx

    ret

global wenda_insert_closest_l2_periodic_avx2
wenda_insert_closest_l2_periodic_avx2:
    %assign indices_buffer_offset 0
    %assign distances_buffer_offset 32
    %assign stack_size 64

    push rbx
    push r12
    push rbp
    mov rbp, rsp

    ; allocate stack and align to 32-byte boundary
    sub rsp, stack_size
    and rsp, -32

    vbroadcastss ymm2, xmm0 ; Load box size
    insert_closest_m compute_l2_periodic, compute_l2_periodic_single, prepare_tail_loop_periodic, .done
.done:
    vzeroupper
    leave
    pop r12
    pop rbx
    ret

align 16
static query_mask, flt_max
query_mask:
    dd -1
    dd -1
    dd -1
    dd 0
flt_max:
    dd 07f7fffffh
