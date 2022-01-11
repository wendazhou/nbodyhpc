.intel_syntax noprefix
.altmacro


# Macro implementation of updating root element in a tournament tree.
#   This implementation makes use of a branchless strategy for better performance.
.macro tournament_tree_update_root_m reg_tree = rdi, reg_idx = rsi, reg_element_idx = rdx, reg_tmp1 = r8, reg_tmp2 = r9
    LOCAL loop_start, finish

    # Load initial index of element
    mov \reg_tmp1, \reg_idx
    shl \reg_tmp1, 32
    or \reg_element_idx, \reg_tmp1

    cmp \reg_idx, 1
    jbe finish
loop_start:
    shr \reg_idx, 1

    # Compute address at rdi + 12 * rsi
    lea rax, [\reg_idx + 2 * \reg_idx]
    lea rax, [\reg_tree + 4 * rax]

    vmovss xmm1, DWORD PTR [rax]
    ucomiss xmm1, xmm0

    # Branchless assign xmm0 to be winner, xmm9 to be loser
    vminss xmm9, xmm0, xmm1
    vmaxss xmm0, xmm0, xmm1
    vmovaps xmm1, xmm9

    # Store current winner as loser, load stored winner
    mov \reg_tmp1, QWORD PTR[rax + 4]
    mov \reg_tmp2, \reg_tmp1
    cmova \reg_tmp2, \reg_element_idx
    mov QWORD PTR [rax + 4], \reg_tmp2
    cmova \reg_element_idx, \reg_tmp1

    vmovss DWORD PTR [rax], xmm1

    cmp \reg_idx, 1
    ja loop_start
finish:
    vmovss DWORD PTR[\reg_tree], xmm0
    mov QWORD PTR[\reg_tree + 4], \reg_element_idx
.endm

# Macro implementation of swapping with top element in the tree.
#    This macro reads out the index of the top element from the top of the tree,
#    and then writes the given values (in fv_reg, sv_reg) to that element's location.
.macro tournament_tree_swap_top_m reg_tree, reg_out_idx, fv_reg, sv_reg, reg_out_address
    mov \reg_out_idx, DWORD PTR [\reg_tree + 8]
    lea \reg_out_address, [\reg_out_idx + \reg_out_idx * 2]
    lea \reg_out_address, [\reg_tree + \reg_out_address * 4]
    vmovss DWORD PTR[\reg_out_address], \fv_reg
    mov DWORD PTR[\reg_out_address + 4], \sv_reg
.endm

# Implementation of updating root element in tournament tree
#   Specialized for a tournament tree of float, uint32_t pairs
#   with comparison only induced by the first element.
#
# Arguments are expected as follows:
#   rdi: address of tournament tree
#   rsi: index at which element is placed
#   xmm0: first value of the inserted element
#   edx: second value of the inserted element
.type tournament_tree_update_root, @function
.global tournament_tree_update_root
tournament_tree_update_root:
    tournament_tree_update_root_m rdi, rsi, rdx, r8, r9
    vzeroupper
    ret


# Implementation of replacing top element in tournament tree
# Arguments are expected as follows:
#   rdi: address of tournament tree
#   xmm0: first value of the inserted element
#   esi: second value of the inserted element
.type tournament_tree_replace_top, @function
.global tournament_tree_replace_top
tournament_tree_replace_top:
    mov edx, esi
    tournament_tree_swap_top_m rdi, esi, xmm0, edx, rax
    jmp tournament_tree_update_root # tail call

.macro transpose_registers
    # Transpose loaded values
    vshufps ymm8, ymm3, ymm4, 0x044
    vshufps ymm9, ymm3, ymm4, 0x0ee
    vshufps ymm10, ymm5, ymm6, 0x044
    vshufps ymm11, ymm5, ymm6, 0x0ee

    vshufps ymm3, ymm8, ymm10, 0x088
    vshufps ymm4, ymm8, ymm10, 0x0dd
    vshufps ymm5, ymm9, ymm11, 0x088
    vshufps ymm7, ymm9, ymm11, 0x0dd
.endm

.macro compute_l2 out_reg = ymm3
    transpose_registers

    # Compute differences
    vsubps ymm3, ymm3, ymm12
    vsubps ymm4, ymm4, ymm13
    vsubps ymm5, ymm5, ymm14

    # Compute squares
    vmulps ymm3, ymm3, ymm3
    vmulps ymm4, ymm4, ymm4
    vmulps ymm5, ymm5, ymm5

    # Compute sum
    vaddps ymm3, ymm3, ymm4
    vaddps \out_reg, ymm3, ymm5
.endm

.macro compute_l2_periodic
    transpose_registers

    # Compute differences
    vsubps ymm3, ymm3, ymm12
    vsubps ymm4, ymm4, ymm13
    vsubps ymm5, ymm5, ymm14

    # Compute squares (including periodic),
    # and select smallest value
.irp reg,ymm3,ymm4,ymm5
    vsubps ymm8, \reg, ymm2
    vaddps ymm9, \reg, ymm2
    vmulps \reg, \reg, \reg
    vmulps ymm8, ymm8, ymm8
    vmulps ymm9, ymm9, ymm9
    vminps \reg, \reg, ymm8
    vminps \reg, \reg, ymm9
.endr

    # Compute sum
    vaddps ymm3, ymm3, ymm4
    vaddps ymm3, ymm3, ymm5
.endm

.macro compute_l2_single out_reg = xmm0
    vmovups \out_reg, XMMWORD PTR [rdi]
    vsubps \out_reg, \out_reg, xmm2
    vdpps \out_reg, \out_reg, \out_reg, 0b01110001
.endm

.macro compute_l2_periodic_single
    vmaskmovps xmm0, xmm8, XMMWORD PTR [rdi]
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
.endm

.macro load_query_vector
    vbroadcastss ymm12, DWORD PTR [rdx]
    vbroadcastss ymm13, DWORD PTR [rdx + 4]
    vbroadcastss ymm14, DWORD PTR [rdx + 8]
.endm

.macro prepare_tail_loop
    vmovaps xmm5, xmm15
    vunpckhps xmm2, xmm12, xmm13
    vxorps xmm13, xmm13, xmm13
    vblendps xmm2, xmm2, xmm14, 0b0100
    vblendps xmm2, xmm2, xmm13, 0b1000
.endm

.macro prepare_tail_loop_periodic
    vmovaps xmm6, xmm2
    vmovaps xmm8, XMMWORD PTR[rip + query_mask]
    prepare_tail_loop
.endm


.type wenda_insert_closest_l2_avx2, @function
.global wenda_insert_closest_l2_avx2
wenda_insert_closest_l2_avx2:
    indices_buffer_offset = 0
    distances_buffer_offset = 32
    stack_size = 64 + 16

    push rbx
    push r12
    push rbp
    mov rbp, rsp

    # allocate stack and align to 32-byte boundary
    sub rsp, stack_size
    and rsp, -32

    load_query_vector

    # Load current best distance from tree
    vbroadcastss ymm15, DWORD PTR [rcx]

    # Adjust rsi to point to end (adjusted for unroll size)
    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

    # Check if main loop required
    cmp rdi, rsi
    jae 4f

    # Load first iteration data
    vmovups ymm3, YMMWORD PTR [rdi]
    vmovups ymm4, YMMWORD PTR [rdi + 32]
    vmovups ymm5, YMMWORD PTR [rdi + 64]
    vmovups ymm6, YMMWORD PTR [rdi + 96]

0:
    # Compute l2 distance for each pair, store results in ymm11. Indices are also extracted and stored in ymm7.
    compute_l2 ymm0

    vcmpltps ymm8, ymm0, ymm15 # Compute pointwise comparison of distances to current best
    vmovmskps ebx, ymm8 # Store comparison result in ebx

    lea rdi, [rdi + 128] # Compute pointer for next iteration
    cmp rdi, rsi
    jae 6f # Skip loading next iteration

    # Load next iteration data
    vmovups ymm3, YMMWORD PTR [rdi]
    vmovups ymm4, YMMWORD PTR [rdi + 32]
    vmovups ymm5, YMMWORD PTR [rdi + 64]
    vmovups ymm6, YMMWORD PTR [rdi + 96]
6:
    test ebx, ebx
    je 1f # skip scalar updates if all elements are worse than current best

    vmovaps YMMWORD PTR[rsp + distances_buffer_offset], ymm0 # Save computed distances to stack
    vmovdqa YMMWORD PTR [rsp + indices_buffer_offset], ymm7 # Save extracted indices to stack
    xor r10, r10
2:  # Start scalar loop
    test ebx, 1
    je 3f

    # Load distance from stack
    vmovss xmm0, DWORD PTR [rsp + distances_buffer_offset + r10]
    ucomiss xmm0, xmm15
    jae 3f

    mov r11d, DWORD PTR[rsp + indices_buffer_offset + r10]
    tournament_tree_swap_top_m rcx, r12d, xmm0, r11d, rax
    tournament_tree_update_root_m rcx, r12, r11, r8, r9

    vbroadcastss ymm15, xmm0
3:  # Test scalar loop
    shr ebx, 1
    add r10, 4
    test ebx, ebx
    jne 2b
1:  # Test main vectorized loop
    cmp rdi, rsi
    jb 0b

4:  # Tail handling start
    add rsi, 7 * 16 # Adjust rsi to point to actual end
    # Check if tail handling required
    cmp rdi, rsi
    je 5f

    prepare_tail_loop
0:  # Tail loop
    vmovups xmm0, XMMWORD PTR [rdi]
    vsubps xmm0, xmm0, xmm2
    vdpps xmm0, xmm0, xmm0, 0b01110001
    ucomiss xmm0, xmm5
    jae 1f

    mov ebx, DWORD PTR[rdi + 12]
    tournament_tree_swap_top_m rcx, r12d, xmm0, ebx, r11
    tournament_tree_update_root_m rcx, r12, rbx, r8, r9

    vmovaps xmm5, xmm0
1:  # Tail loop check
    add rdi, 16
    cmp rdi, rsi
    jb 0b

5:
    vzeroupper
    leave
    pop r12
    pop rbx

    ret

.macro find_closest_m compute_distance, compute_distance_single, prepare_tail, epilog_label
    load_query_vector
    vbroadcastss ymm15, DWORD PTR [rip + flt_max]

    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

    # Check if main loop required
    cmp rdi, rsi
    jae 4f
0:
    vmovups ymm3, YMMWORD PTR [rdi]
    vmovups ymm4, YMMWORD PTR [rdi + 32]
    vmovups ymm5, YMMWORD PTR [rdi + 64]
    vmovups ymm6, YMMWORD PTR [rdi + 96]

    compute_distance

    vcmpltps ymm8, ymm3, ymm15
    vmovmskps eax, ymm8
    test eax, eax
    je 1f

    vmovdqa YMMWORD PTR [rsp + 32], ymm7
    vmovaps YMMWORD PTR [rsp], ymm3
    xor r10, r10
2:
    test eax, 1
    je 3f

    vmovss xmm0, DWORD PTR [rsp + 4 * r10]
    ucomiss xmm0, xmm15
    jae 3f

    vbroadcastss ymm15, xmm0
    mov r11d, [rsp + 32 + 4 * r10]
3:
    shr eax, 1
    inc r10
    test eax, eax
    jne 2b
1:
    lea rdi, [rdi + 128]
    cmp rdi, rsi
    jb 0b

    mov eax, r11d

4: # Tail handling start
    add rsi, 7 * 16 # Adjust end pointer to non-truncated value
    cmp rdi, rsi # Test if tail loop required
    je epilog_label # epilog

    prepare_tail
0:
    compute_distance_single
    ucomiss xmm0, xmm5
    jae 1f

    vmovaps xmm5, xmm0
    mov eax, DWORD PTR[rdi + 12]
1:
    add rdi, 16
    cmp rdi, rsi
    jb 0b

.endm

.type wenda_find_closest_l2_avx2, @function
.global wenda_find_closest_l2_avx2
wenda_find_closest_l2_avx2:
    push rbp
    mov rbp, rsp
    sub rsp, 64
    and rsp, -32

    find_closest_m compute_l2, compute_l2_single, prepare_tail_loop, 5f

5:
    vzeroupper
    leave
    ret

.type wenda_find_closest_l2_periodic_avx2, @function
.global wenda_find_closest_l2_periodic_avx2
wenda_find_closest_l2_periodic_avx2:
    push rbp
    mov rbp, rsp
    sub rsp, 64
    and rsp, -32

    vbroadcastss ymm2, xmm0
    find_closest_m compute_l2_periodic, compute_l2_periodic_single, prepare_tail_loop_periodic, 5f
5:
    vzeroupper
    leave
    ret

.align 16
query_mask:
    .long -1
    .long -1
    .long -1
    .long 0
flt_max:
    .long 0x07f7fffff

