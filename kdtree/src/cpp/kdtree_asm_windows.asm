.code
; Implementation of replacing top element in tournament tree
;   Specialized for a tournament tree of float, uint32_t pairs
;   with comparison only induced by the first element.
;
; Arguments are expected as follows:
;   rcx: address of tournament tree
;   edx: index at which element is placed
;   xmm0: value of the inserted element
;   r9d: index of the inserted element
tournament_tree_update_root PROC PUBLIC
    cmp edx, 1
    jbe finish

    ; Load initial index of element into ecx
    mov r10d, edx

    ; In the main loop, we maintain a copy of the current value,
    ; which is comprised of a triplet (float, uint32_t, uint32_t),
    ; with the first two values corresponding to the value stored
    ; in the tournament tree, and the last value corresponding to
    ; the original index of the element in the tournament tree.
    ; They are kept in registers xmm0, r9d, r10d
loop_start:
    shr edx, 1

    ; Compute address at rcx + 12 * rdx
    lea rax, [edx + 2 * edx]
    lea rax, [rcx + 4 * rax]

    movss xmm1, DWORD PTR [rax]
    ucomiss xmm1, xmm0
    jbe loop_check

    ; Store current winner as loser, load stored winner
    mov r11d, DWORD PTR[rax + 4]
    mov DWORD PTR [rax + 4], r9d
    mov r8d, DWORD PTR[rax + 8]
    mov DWORD PTR [rax + 8], r10d
    mov r9d, r11d
    mov r10d, r8d

    movss DWORD PTR [rax], xmm0
    movss xmm0, xmm1

loop_check:
    cmp edx, 1
    ja loop_start
finish:
    movss DWORD PTR[rcx], xmm0
    mov DWORD PTR[rcx + 4], r9d
    mov DWORD PTR[rcx + 8], r10d
    ret
tournament_tree_update_root ENDP
END
