.intel_syntax noprefix
.altmacro


# Macro implementation of updating root element in a tournament tree.
#   This implementation makes use of a branchless strategy for better performance.
.macro tournament_tree_update_root_m reg_tree = rdi, reg_idx = rsi, reg_element_idx = rdx, reg_tmp1 = r8, reg_tmp2 = r9
    LOCAL loop_start, finish
    cmp \reg_idx, 1
    jbe finish

    # Load initial index of element
    mov \reg_tmp1, \reg_idx
    shl \reg_tmp1, 32
    or \reg_element_idx, \reg_tmp1

loop_start:
    shr \reg_idx, 1

    # Compute address at rdi + 12 * rsi
    lea rax, [\reg_idx + 2 * \reg_idx]
    lea rax, [\reg_tree + 4 * rax]

    movss xmm1, DWORD PTR [rax]
    ucomiss xmm1, xmm0

    # Branchless assign xmm0 to be winner, xmm9 to be loser
    vminss xmm9, xmm0, xmm1
    vmaxss xmm0, xmm0, xmm1
    movss xmm1, xmm9

    # Store current winner as loser, load stored winner
    mov \reg_tmp1, QWORD PTR[rax + 4]
    mov \reg_tmp2, \reg_tmp1
    cmova \reg_tmp2, \reg_element_idx
    mov QWORD PTR [rax + 4], \reg_tmp2
    cmova \reg_element_idx, \reg_tmp1

    movss DWORD PTR [rax], xmm1

    cmp \reg_idx, 1
    ja loop_start
finish:
    movss DWORD PTR[\reg_tree], xmm0
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

.macro load_and_compute_l2
    vmovups ymm3, YMMWORD PTR [rdi]
    vmovups ymm4, YMMWORD PTR [rdi + 32]
    vmovups ymm5, YMMWORD PTR [rdi + 64]
    vmovups ymm6, YMMWORD PTR [rdi + 96]

    vpunpckhdq ymm7, ymm3, ymm5
    vpunpckhdq ymm8, ymm4, ymm6
    vpunpckhdq ymm7, ymm7, ymm8

    vsubps ymm3, ymm3, ymm2
    vsubps ymm4, ymm4, ymm2
    vsubps ymm5, ymm5, ymm2
    vsubps ymm6, ymm6, ymm2

    vdpps ymm3, ymm3, ymm3, 0b01110001
    vdpps ymm4, ymm4, ymm4, 0b01110010
    vdpps ymm5, ymm5, ymm5, 0b01110100
    vdpps ymm6, ymm6, ymm6, 0b01111000

    vaddps ymm3, ymm3, ymm4
    vaddps ymm5, ymm5, ymm6
    vaddps ymm3, ymm3, ymm5
.endm

.macro load_query_vector
    vmovdqa xmm2, XMMWORD PTR [query_mask]
    vmaskmovps xmm2, xmm2, XMMWORD PTR [rdx]
    vinsertf128 ymm2, ymm2, xmm2, 1
.endm

.type wenda_insert_closest_l2_avx2, @function
.global wenda_insert_closest_l2_avx2
wenda_insert_closest_l2_avx2:
    indices_buffer_offset = 0
    distances_buffer_offset = 32

    push rbx
    push r12

    sub rsp, 64

    load_query_vector

    # Load current best distance from tree
    vbroadcastss ymm10, DWORD PTR [rcx]

    # Adjust rsi to point to end (adjusted for unroll size)
    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

0:
    load_and_compute_l2
    vmovups YMMWORD PTR [rsp + indices_buffer_offset], ymm7

    vcmpltps ymm7, ymm3, ymm10
    vmovmskps ebx, ymm7
    test ebx, ebx
    je 1f

    vmovups YMMWORD PTR[rsp + distances_buffer_offset], ymm3
    xor r10, r10

2:
    test ebx, 1
    je 3f

    vmovss xmm0, DWORD PTR [rsp + distances_buffer_offset + 4 * r10]
    ucomiss xmm0, xmm10
    jae 3f

    mov r11d, DWORD PTR[rsp + indices_buffer_offset + 4 * r10]
    tournament_tree_swap_top_m rcx, r12d, xmm0, r11d, rax
    tournament_tree_update_root_m rcx, r12, r11, r8, r9

    vbroadcastss ymm10, xmm0
3:
    shr ebx, 1
    inc r10
    test ebx, ebx
    jne 2b

1:
    lea rdi, [rdi + 128]
    cmp rdi, rsi
    jb 0b

    add rsp, 64

    pop r12
    pop rbx

    ret

.type wenda_find_closest_l2_avx2, @function
.global wenda_find_closest_l2_avx2
wenda_find_closest_l2_avx2:
    sub rsp, 64

    load_query_vector
    vbroadcastss ymm10, DWORD PTR [flt_max]

    lea rsi, [rsi * 8]
    lea rsi, [rdi + rsi * 2 - 7 * 16]

0:
    load_and_compute_l2
    vmovups YMMWORD PTR [rsp + 32], ymm7

    vcmpltps ymm7, ymm3, ymm10
    vmovmskps eax, ymm7
    test eax, eax
    je 1f

    vmovups YMMWORD PTR [rsp], ymm3
    xor r10, r10

2:
    test eax, 1
    je 3f

    vmovss xmm0, DWORD PTR [rsp + 4 * r10]
    ucomiss xmm0, xmm10
    jae 3f

    vbroadcastss ymm10, xmm0
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

# Adjust end pointer to non-truncated value
    add rsi, 7 * 16
# Test for tail loop
    cmp rdi, rsi
    je .done

# Tail loop
    vmovaps xmm5, xmm10
.tail_start:
    vmovups xmm3, XMMWORD PTR [rdi]
    vsubps xmm3, xmm3, xmm2
    vdpps xmm3, xmm3, xmm3, 0b01110001
    ucomiss xmm3, xmm5
    jae .tail_end

    vbroadcastss xmm5, xmm3
    mov eax, DWORD PTR[rdi + 12]
.tail_end:
    add rdi, 16
    cmp rdi, rsi
    jb .tail_start

.done:
    add rsp, 64
    ret

.align 16
query_mask:
    .long -1
    .long -1
    .long -1
    .long 0
flt_max:
    .long 0x07f7fffff

