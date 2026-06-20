    .option norvc

    .equ SIG_BASE,        0x10008000
    .equ SIG_STATUS,      0x10008000
    .equ SIG_PASS_COUNT,  0x10008004
    .equ SIG_FAIL_TEST,   0x10008008
    .equ SIG_FAIL_INDEX,  0x1000800c
    .equ SIG_FAIL_GOT,    0x10008010
    .equ SIG_FAIL_EXP,    0x10008014

    .equ SRC_A,           0x10100000
    .equ SRC_B,           0x10100100
    .equ DST_BASE,        0x10100200
    .equ DST_ADD,         0x10100200
    .equ DST_SUB,         0x10100300
    .equ DST_AND,         0x10100400
    .equ DST_OR,          0x10100500
    .equ DST_XOR,         0x10100600
    .equ DST_SLL,         0x10100700
    .equ DST_SRL,         0x10100800
    .equ VL,              16

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

    li s1, SRC_A
    li s2, SRC_B
    li t0, 0
1:
    addi t1, t0, 1
    addi t2, t0, 16
    sw t1, 0(s1)
    sw t2, 0(s2)
    addi s1, s1, 4
    addi s2, s2, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b

    li t0, VL
    vsetvli t1, t0, e32, m1, ta, ma

    li a0, SRC_A
    vle32.v v1, (a0)
    li a0, SRC_B
    vle32.v v2, (a0)

    vadd.vv v3, v1, v2
    li a1, DST_ADD
    vse32.v v3, (a1)
    fence
    li a0, DST_ADD
    li a1, 1
    jal ra, verify_add
    jal ra, mark_pass

    vsub.vv v3, v1, v2
    li a1, DST_SUB
    vse32.v v3, (a1)
    fence
    li a0, DST_SUB
    li a1, 2
    jal ra, verify_sub
    jal ra, mark_pass

    vand.vv v3, v1, v2
    li a1, DST_AND
    vse32.v v3, (a1)
    fence
    li a0, DST_AND
    li a1, 3
    jal ra, verify_and
    jal ra, mark_pass

    vor.vv v3, v1, v2
    li a1, DST_OR
    vse32.v v3, (a1)
    fence
    li a0, DST_OR
    li a1, 4
    jal ra, verify_or
    jal ra, mark_pass

    vxor.vv v3, v1, v2
    li a1, DST_XOR
    vse32.v v3, (a1)
    fence
    li a0, DST_XOR
    li a1, 5
    jal ra, verify_xor
    jal ra, mark_pass

    vsll.vi v3, v1, 1
    li a1, DST_SLL
    vse32.v v3, (a1)
    fence
    li a0, DST_SLL
    li a1, 6
    jal ra, verify_sll
    jal ra, mark_pass

    vsrl.vi v3, v2, 1
    li a1, DST_SRL
    vse32.v v3, (a1)
    fence
    li a0, DST_SRL
    li a1, 7
    jal ra, verify_srl
    jal ra, mark_pass

    li t0, 0xdeadbeef
    li t1, SIG_STATUS
    sw t0, 0(t1)
done:
    j done

mark_pass:
    li t0, SIG_PASS_COUNT
    lw t1, 0(t0)
    addi t1, t1, 1
    sw t1, 0(t0)
    ret

verify_add:
    li t0, 0
1:
    lw t1, 0(a0)
    slli t2, t0, 1
    addi t2, t2, 17
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
    ret

verify_sub:
    li t0, 0
1:
    lw t1, 0(a0)
    li t2, -15
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
    ret

verify_and:
    li t0, 0
1:
    lw t1, 0(a0)
    addi t2, t0, 1
    addi t3, t0, 16
    and t2, t2, t3
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

verify_or:
    li t0, 0
1:
    lw t1, 0(a0)
    addi t2, t0, 1
    addi t3, t0, 16
    or t2, t2, t3
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

verify_xor:
    li t0, 0
1:
    lw t1, 0(a0)
    addi t2, t0, 1
    addi t3, t0, 16
    xor t2, t2, t3
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t4, VL
    blt t0, t4, 1b
    ret

verify_sll:
    li t0, 0
1:
    lw t1, 0(a0)
    addi t2, t0, 1
    slli t2, t2, 1
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
    ret

verify_srl:
    li t0, 0
1:
    lw t1, 0(a0)
    addi t2, t0, 16
    srli t2, t2, 1
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b
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
1:
    j 1b
