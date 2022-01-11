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

; Loads and computes distances (L2) from rcx
; Distances are saved to ymm3, and indices are saved to ymm7.
compute_l2 MACRO
    vmovups ymm3, YMMWORD PTR [rcx]
    vmovups ymm4, YMMWORD PTR [rcx + 32]
    vmovups ymm5, YMMWORD PTR [rcx + 64]
    vmovups ymm6, YMMWORD PTR [rcx + 96]

    vpunpckhdq ymm7, ymm3, ymm5
    vpunpckhdq ymm8, ymm4, ymm6
    vpunpckhdq ymm7, ymm7, ymm8

    vsubps ymm3, ymm3, ymm2
    vsubps ymm4, ymm4, ymm2
    vsubps ymm5, ymm5, ymm2
    vsubps ymm6, ymm6, ymm2

    vdpps ymm3, ymm3, ymm3, 01110001b
    vdpps ymm4, ymm4, ymm4, 01110010b
    vdpps ymm5, ymm5, ymm5, 01110100b
    vdpps ymm6, ymm6, ymm6, 01111000b

    vaddps ymm3, ymm3, ymm4
    vaddps ymm5, ymm5, ymm6
    vaddps ymm3, ymm3, ymm5
ENDM

; Loads and computes distances (L2) from rcx
; Distances are saved to ymm3, and indices are saved to ymm7.
; This function does not use horizontal dot product, but rather
; transposes in registers.
compute_l2_transpose MACRO query_reg
    vmovups ymm3, YMMWORD PTR [rcx]
    vmovups ymm4, YMMWORD PTR [rcx + 32]
    vmovups ymm5, YMMWORD PTR [rcx + 64]
    vmovups ymm6, YMMWORD PTR [rcx + 96]

    ; Transpose loaded values
    vshufps ymm8, ymm3, ymm4, 044h
    vshufps ymm9, ymm3, ymm4, 0eeh
    vshufps ymm10, ymm5, ymm6, 044h
    vshufps ymm11, ymm5, ymm6, 0eeh

    vshufps ymm3, ymm8, ymm10, 088h
    vshufps ymm4, ymm8, ymm10, 0ddh
    vshufps ymm5, ymm9, ymm11, 088h
    vshufps ymm7, ymm9, ymm11, 0ddh

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
    vaddps ymm3, ymm3, ymm5
ENDM

; Computes the l2 distance to the query
; for a single point, loaded from rcx.
compute_l2_single MACRO query_reg
    vmovaps xmm0, XMMWORD PTR [rcx]
    vsubps xmm0, xmm0, query_reg
    vdpps xmm0, xmm0, xmm0, 01110001b
ENDM

; Loads query vector and duplicates across lanes
; This load is associated with the compute_l2 methods.
load_query_vector MACRO reg_source
    vmovdqa xmm2, XMMWORD PTR [query_mask]
    vmaskmovps xmm2, xmm2, XMMWORD PTR [reg_source]
    vinsertf128 ymm2, ymm2, xmm2, 1
ENDM

; Loads query vector, and broadcasts to ymm12, ymm13, ymm14
; by component. Also loads the query vector into ymm2 for
; compatibility with scalar compute methods.
load_query_vector_transpose MACRO reg_source
    vbroadcastss ymm12, DWORD PTR [reg_source]
    vbroadcastss ymm13, DWORD PTR [reg_source + 4]
    vbroadcastss ymm14, DWORD PTR [reg_source + 8]

    ; Unpack query vector into ymm2 for compatibility with scalar computation
    vunpckhps ymm2, ymm12, ymm13
    vblendps ymm2, ymm2, ymm14, 01000100b
ENDM

; Compares all distances to stored value in max_reg, and writes out result to mask_reg
; If any are smaller, stores distances and indices to stack.
; Otherwise, jumps to loop_end label
compare_distances MACRO mask_reg, max_reg
    vcmpltps ymm8, ymm3, max_reg
    vmovmskps mask_reg, ymm8
    test mask_reg, mask_reg
    je loop_end

    vmovdqa YMMWORD PTR [indices_buffer], ymm7
    vmovaps YMMWORD PTR [distances_buffer], ymm3
    xor r10, r10
ENDM

; Tests whether scalar operations are required for this iteration.
;    This macro checks whether the mask_reg bit is set, and if so,
;    loads the computed distance and checks whether it is smaller
;    than the current best. If any checks fail, it jumps to end_label.
scalar_insert_test MACRO mask_reg, max_reg, end_label
    test mask_reg, 1
    je end_label

    ; perform scalar insertion
    vmovss xmm0, DWORD PTR [distances_buffer + r10]
    ucomiss xmm0, max_reg
    jae end_label
ENDM

; Tests whether there are any remaining scalar elements to process,
; and if so, shifts the mask register, increments r10, and jumps to start_label.
scalar_insert_loop MACRO mask_reg, start_label
    shr mask_reg, 1
    add r10, 4
    test mask_reg, mask_reg
    jne start_label
ENDM

; Macro for find closest function, parametrized by the distance computation
; to allow for different distance parametrizations.
find_closest_m MACRO load_query, compute_distance, compute_distance_single
    current_max_reg EQU ymm15
    current_max_reg_xmm EQU xmm15

    ; Load query vector into xmm0, and duplicate across lanes
    load_query r8

    ; Load current best distance
    vbroadcastss current_max_reg, DWORD PTR [flt_max]

    ; Load end pointer into rdx
    lea rdx, [rdx * 8]
    lea rdx, [rcx + rdx * 2 - 7 * 16]

    ; Check if any main iterations need to be done
    cmp rcx, rdx
    jae tail_loop
loop_start:
    compute_distance
    compare_distances eax, current_max_reg
scalar_insert_start:
    scalar_insert_test eax, current_max_reg_xmm, scalar_insert_end

    vbroadcastss current_max_reg, xmm0
    mov r11d, [indices_buffer + r10]
scalar_insert_end:
    scalar_insert_loop eax, scalar_insert_start
loop_end:
    lea rcx, [rcx + 128]
    cmp rcx, rdx
    jb loop_start

; Store best result so far into eax
    mov eax, r11d

tail_loop:
; Adjust end pointer to non-truncated value
    add rdx, 7 * 16
; Test for any iterations in tail loop
    cmp rcx, rdx
    je done
; Tail loop (remainder)
    vmovaps xmm5, current_max_reg_xmm
tail_loop_start:
    compute_distance_single xmm2
    ucomiss xmm0, xmm5
    jae tail_end

    vmovaps xmm5, xmm0
    mov eax, DWORD PTR[rcx + 12]
tail_end:
    add rcx, 16
    cmp rcx, rdx
    jb tail_loop_start
ENDM


; Procedure for finding the closest element in positions to given query
; C prototype: void find_closest_element(const char* positions, size_t n, const float* query)
;
; Arguments:
;   - rcx: pointer to the positions array
;   - rdx: length of positions array
;   - r8: pointer to query vector
wenda_find_closest_l2_avx2 PROC PUBLIC FRAME
    ; Stack-based variables
    stack_size = 64 + 10 * 16 + 16

    push rbp
.pushreg rbp
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
.setframe rsp, 0
    save_xmm_registers rsp, 64 + 16, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15
.endprolog

    ; Align rbp to 32-byte boundary.
    lea rbp, [rsp + 16]
    and rbp, -32

    indices_buffer EQU rbp
    distances_buffer EQU rbp + 32

    find_closest_m load_query_vector_transpose, compute_l2_transpose, compute_l2_single
done:
; Epilog
    restore_xmm_registers rsp, 64 + 16, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15
    vzeroupper
    add rsp, stack_size
    pop rbp
    ret

wenda_find_closest_l2_avx2 ENDP

; Main procedure for finding k closest points to query vector,
;
; Arguments:
;   - rcx: address of array of points, index pairs
;   - rdx: number of points in array
;   - r8: address of query vector
;   - r9: address of the tournament tree
; Returns:
;   - rax: Index of the closest point
wenda_insert_closest_l2_avx2 PROC PUBLIC FRAME
    stack_size = 64 + 4 * 16 + 16

    FOR reg, <rbp,rbx,rdi,rsi,r12,r13,r14>
        push reg
        .pushreg reg
    ENDM
    sub rsp, stack_size ; reserve space for local variables and non-volatile registers
.allocstack stack_size
.setframe rsp, 0
    save_xmm_registers rsp, 64 + 16, xmm6, xmm7, xmm8, xmm9
.endprolog

    ; Align rbp to 32-byte boundary
    lea rbp, [rsp + 16]
    and rbp, -32

    indices_buffer EQU rbp
    distances_buffer EQU rbp + 32

    load_query_vector r8

    ; Load current best distance from tree
    vbroadcastss ymm9, DWORD PTR [r9]

    ; Load end pointer into rdx
    lea rdx, [rdx * 8]
    lea rdx, [rcx + rdx * 2 - 7 * 16]

    cmp rcx, rdx
    jae tail_loop

loop_start:
    compute_l2
    compare_distances ebx, ymm9
scalar_insert_start:
    scalar_insert_test ebx, xmm9, scalar_insert_end

    ; update tournament tree with new best
    mov esi, [indices_buffer + r10]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_branchless_m r9, rdi, rsi, r11, r12, xmm3

    ; reload top value, xmm0 is updated by the tournament tree macros
    vbroadcastss ymm9, xmm0
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
    vmovaps xmm5, xmm9
tail_loop_start:
    vmovaps xmm0, XMMWORD PTR [rcx]
    vsubps xmm0, xmm0, xmm2
    vdpps xmm0, xmm0, xmm0, 01110001b
    ucomiss xmm0, xmm5
    jae tail_end

    mov esi, DWORD PTR[rcx + 12]
    tournament_tree_swap_top_m r9, edi, xmm0, esi, r11
    tournament_tree_update_root_branchless_m r9, rdi, rsi, r11, r12, xmm3

    vbroadcastss xmm5, xmm0
tail_end:
    add rcx, 16
    cmp rcx, rdx
    jb tail_loop_start

done:
; epilog
    restore_xmm_registers rsp, 64 + 16, xmm6, xmm7, xmm8, xmm9
    vzeroupper
    add rsp, stack_size
    FOR reg, <r14,r13,r12,rsi,rdi,rbx,rbp>
        pop reg
    ENDM
    ret
wenda_insert_closest_l2_avx2 ENDP

ALIGN 16
query_mask DWORD -1, -1, -1, 0
flt_max DWORD 07f7fffffr

END
