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
