.intel_syntax noprefix

# Implementation of replacing top element in tournament tree
#   Specialized for a tournament tree of float, uint32_t pairs
#   with comparison only induced by the first element.
#
# Arguments are expected as follows:
#   rdi: address of tournament tree
#   rsi: index at which element is placed
#   xmm0: value of the inserted element
#   edx: index of the inserted element
.type tournament_tree_update_root, @function
.global tournament_tree_update_root
tournament_tree_update_root:
    cmp esi, 1
    jbe .finish

    # Load initial index of element into ecx
    mov ecx, esi

    # In the main loop, we maintain a copy of the current value,
    # which is comprised of a triplet (float, uint32_t, uint32_t),
    # with the first two values corresponding to the value stored
    # in the tournament tree, and the last value corresponding to
    # the original index of the element in the tournament tree.
    # They are kept in registers xmm0, edx, ecx
.loop:
    shr esi

    # Compute address at rdi + 12 * rsi
    lea r8, [rsi + 2 * rsi]
    lea r8, [rdi + 4 * r8]

    movss xmm1, DWORD PTR [r8]
    ucomiss xmm1, xmm0
    jbe .loop_check

    # Store current winner as loser, load stored winner
    mov eax, DWORD PTR[r8 + 4]
    mov DWORD PTR [r8 + 4], edx
    mov r9d, DWORD PTR[r8 + 8]
    mov DWORD PTR [r8 + 8], ecx
    mov edx, eax
    mov ecx, r9d

    movss DWORD PTR [r8], xmm0
    movss xmm0, xmm1

.loop_check:
    cmp esi, 1
    ja .loop
.finish:
    movss DWORD PTR[rdi], xmm0
    mov DWORD PTR[rdi + 4], edx
    mov DWORD PTR[rdi + 8], ecx
    ret

.type insert_shorter_distance_avx2, @function
.global insert_shorter_distance_avx2
insert_shorter_distance_avx2:
    sub rsp, 64

    vmovdqa xmm2, XMMWORD PTR [query_mask]
    vmaskmovps xmm2, xmm2, XMMWORD PTR [rdx]
    vinsertf128 ymm2, ymm2, xmm2, 1
    vbroadcastss ymm10, DWORD PTR [flt_max]

    lea rsi, [rsi  * 8]
    lea rsi, [rdi + rsi * 8]

.loop_start:
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

    vmovups YMMWORD PTR [rsp + 32], ymm7

    vcmpltps ymm7, ymm3, ymm10
    vmovmskps eax, ymm7
    test eax, eax
    je .loop_end

    vmovups YMMWORD PTR [rsp], ymm3
    xor r10, r10

.scalar_insert_start:
    test eax, 1
    je .scalar_insert_end

    vmovss xmm0, DWORD PTR [rsp + 4 * r10]
    ucomiss xmm0, xmm10
    jae .scalar_insert_end

    vbroadcastss ymm10, xmm0
    mov r11d, [rsp + 32 + 4 * r10]
.scalar_insert_end:
    shr eax, 1
    inc r10
    test eax, eax
    jne .scalar_insert_start
.loop_end:
    lea rdi, [rdi + 128]
    cmp rdi, rsi
    jb .loop_start

    mov eax, r11d
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

