    .option norvc
    # Scenario: RVV non-unit memory regression for Spatz VLSU.
    # Target: e8 strided load/store plus unordered and ordered indexed load/store.
    # These instructions are required before using Spatz for im2col gather/scatter.

    .equ SIG_BASE,        0x10008000
    .equ SIG_STATUS,      0x10008000
    .equ SIG_PASS_COUNT,  0x10008004
    .equ SIG_FAIL_TEST,   0x10008008
    .equ SIG_FAIL_INDEX,  0x1000800c
    .equ SIG_FAIL_GOT,    0x10008010
    .equ SIG_FAIL_EXP,    0x10008014
    .equ SIG_PHASE,       0x10008018
    .equ SIG_OP,          0x1000801c
    .equ IRQ_HOST_NOTIFY, 0x20002018

    .equ SRC_STRIDED,     0x10104000
    .equ DST_STRIDED,     0x10104100
    .equ SRC_INDEXED,     0x10104200
    .equ SRC_IDX8,        0x10104300
    .equ DST_IDX8,        0x10104400
    .equ DST_VLUXEI,      0x10104500
    .equ DST_VSUXEI,      0x10104600
    .equ SRC_ORDERED,     0x10104700
    .equ ORD_IDX8,        0x10104800
    .equ DST_ORD_IDX8,    0x10104900
    .equ DST_VLOXEI,      0x10104a00
    .equ DST_VSOXEI,      0x10104b00

    .equ VL,              16
    .equ STRIDE_LOAD,     3
    .equ STRIDE_STORE,    2

    .section .text.init
    .global _start

_start:
    la sp, __stack_top
    li s0, SIG_BASE
    sw zero, 0(s0)
    sw zero, 4(s0)
    sw zero, 8(s0)
    sw zero, 12(s0)
    sw zero, 16(s0)
    sw zero, 20(s0)
    sw zero, 24(s0)
    sw zero, 28(s0)

    jal ra, seed_strided
    jal ra, seed_indexed
    jal ra, seed_ordered

    # Test 1: gather every third byte, then scatter every second byte.
    li t0, VL
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, SRC_STRIDED
    li a1, STRIDE_LOAD
    vlse8.v v1, (a0), a1
    li a0, DST_STRIDED
    li a1, STRIDE_STORE
    vsse8.v v1, (a0), a1
    fence
    li a0, DST_STRIDED
    li a1, 1
    jal ra, verify_strided
    jal ra, mark_pass

    # Test 2: unordered indexed load gathers sparse bytes into a dense vector.
    li t0, VL
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, SRC_IDX8
    vle8.v v2, (a0)
    li a0, SRC_INDEXED
    vluxei8.v v1, (a0), v2
    li a0, DST_VLUXEI
    vse8.v v1, (a0)
    fence
    li a0, DST_VLUXEI
    li a1, 2
    jal ra, verify_vluxei
    jal ra, mark_pass

    # Test 3: unordered indexed store scatters the gathered vector.
    li t0, VL
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, DST_IDX8
    vle8.v v2, (a0)
    li a0, DST_VSUXEI
    vsuxei8.v v1, (a0), v2
    fence
    li a0, DST_VSUXEI
    li a1, 3
    jal ra, verify_vsuxei
    jal ra, mark_pass

    # Test 4: ordered indexed load uses a different source/index fixture.
    li t0, VL
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, ORD_IDX8
    vle8.v v2, (a0)
    li a0, SRC_ORDERED
    vloxei8.v v1, (a0), v2
    li a0, DST_VLOXEI
    vse8.v v1, (a0)
    fence
    li a0, DST_VLOXEI
    li a1, 4
    jal ra, verify_vloxei
    jal ra, mark_pass

    # Test 5: ordered indexed store scatters the ordered-load result.
    li t0, VL
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, DST_ORD_IDX8
    vle8.v v2, (a0)
    li a0, DST_VSOXEI
    vsoxei8.v v1, (a0), v2
    fence
    li a0, DST_VSOXEI
    li a1, 5
    jal ra, verify_vsoxei
    jal ra, mark_pass

    li t0, 0xdeadbeef
    li t1, SIG_STATUS
    sw t0, 0(t1)
    li t1, IRQ_HOST_NOTIFY
    sw t0, 0(t1)
done:
    j done

seed_strided:
    li a0, SRC_STRIDED
    li a1, DST_STRIDED
    li t0, 0
1:
    li t1, 5
    mul t1, t1, t0
    addi t1, t1, 7
    sb t1, 0(a0)
    sb zero, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t2, 80
    blt t0, t2, 1b
    ret

seed_indexed:
    li a0, SRC_INDEXED
    li a1, DST_VSUXEI
    li t0, 0
1:
    li t1, 9
    mul t1, t1, t0
    addi t1, t1, 3
    sb t1, 0(a0)
    sb zero, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t2, 96
    blt t0, t2, 1b

    li a0, SRC_IDX8
    li a1, DST_IDX8
    li t0, 0
2:
    li t1, 5
    mul t1, t1, t0
    addi t1, t1, 1
    sb t1, 0(a0)
    slli t2, t0, 1
    sb t2, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 2b
    ret

seed_ordered:
    li a0, SRC_ORDERED
    li a1, DST_VSOXEI
    li t0, 0
1:
    li t1, 7
    mul t1, t1, t0
    addi t1, t1, 11
    sb t1, 0(a0)
    sb zero, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t2, 96
    blt t0, t2, 1b

    li a0, ORD_IDX8
    li a1, DST_ORD_IDX8
    li t0, 0
2:
    li t1, 4
    mul t1, t1, t0
    addi t1, t1, 2
    sb t1, 0(a0)
    li t2, 3
    mul t2, t2, t0
    sb t2, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 2b
    ret

mark_pass:
    li t0, SIG_PASS_COUNT
    lw t1, 0(t0)
    addi t1, t1, 1
    sw t1, 0(t0)
    ret

verify_strided:
    li t0, 0
1:
    slli t3, t0, 1
    add t3, a0, t3
    lbu t1, 0(t3)
    li t2, 15
    mul t2, t2, t0
    addi t2, t2, 7
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

verify_vluxei:
    li t0, 0
1:
    lbu t1, 0(a0)
    li t2, 45
    mul t2, t2, t0
    addi t2, t2, 12
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi a0, a0, 1
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
    ret

verify_vsuxei:
    li t0, 0
1:
    slli t3, t0, 1
    add t3, a0, t3
    lbu t1, 0(t3)
    li t2, 45
    mul t2, t2, t0
    addi t2, t2, 12
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

verify_vloxei:
    li t0, 0
1:
    lbu t1, 0(a0)
    li t2, 28
    mul t2, t2, t0
    addi t2, t2, 25
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi a0, a0, 1
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
    ret

verify_vsoxei:
    li t0, 0
1:
    li t3, 3
    mul t3, t3, t0
    add t3, a0, t3
    lbu t1, 0(t3)
    li t2, 28
    mul t2, t2, t0
    addi t2, t2, 25
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

fail:
    li t3, SIG_FAIL_TEST
    sw a1, 0(t3)
    li t3, SIG_FAIL_INDEX
    sw t0, 0(t3)
    li t3, SIG_FAIL_GOT
    sw t1, 0(t3)
    li t3, SIG_FAIL_EXP
    sw t2, 0(t3)
    li t3, SIG_STATUS
    li t4, 0xbad00000
    or t4, t4, a1
    sw t4, 0(t3)
    li t3, IRQ_HOST_NOTIFY
    sw t4, 0(t3)
1:
    j 1b
