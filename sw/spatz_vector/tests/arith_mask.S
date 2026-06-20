    .option norvc

    .equ SIG_BASE,        0x10008000
    .equ SIG_STATUS,      0x10008000
    .equ SIG_PASS_COUNT,  0x10008004
    .equ SIG_FAIL_TEST,   0x10008008
    .equ SIG_FAIL_INDEX,  0x1000800c
    .equ SIG_FAIL_GOT,    0x10008010
    .equ SIG_FAIL_EXP,    0x10008014
    .equ SIG_PHASE,       0x10008018
    .equ SIG_OP,          0x1000801c

    .equ SRC_A,           0x10102000
    .equ SRC_B,           0x10102100
    .equ DST_MUL,         0x10102200
    .equ DST_MIN,         0x10102300
    .equ DST_MAX,         0x10102400
    .equ DST_SRA,         0x10102500
    .equ DST_MERGE,       0x10102600
    .equ VL,              16

    .section .text.init
    .global _start

_start:
    la sp, __stack_top
    li s0, SIG_BASE
    li t0, 0
    sw t0, 0(s0)
    sw t0, 4(s0)
    sw t0, 8(s0)
    sw t0, 12(s0)
    sw t0, 16(s0)
    sw t0, 20(s0)
    sw t0, 24(s0)
    sw t0, 28(s0)

    li a0, SRC_A
    li a1, SRC_B
    li t0, 0
1:
    li t1, 3
    mul t1, t1, t0
    addi t1, t1, -24
    li t2, 2
    mul t2, t2, t0
    addi t2, t2, -7
    sw t1, 0(a0)
    sw t2, 0(a1)
    addi a0, a0, 4
    addi a1, a1, 4
    addi t0, t0, 1
    li t3, VL
    blt t0, t3, 1b

    li t0, VL
    vsetvli t1, t0, e32, m1, ta, ma
    li a0, SRC_A
    vle32.v v1, (a0)
    li a0, SRC_B
    vle32.v v2, (a0)

    vmul.vv v3, v1, v2
    li a0, DST_MUL
    vse32.v v3, (a0)
    fence
    li a0, DST_MUL
    li a1, 1
    jal ra, verify_mul
    jal ra, mark_pass

    vmin.vv v3, v1, v2
    li a0, DST_MIN
    vse32.v v3, (a0)
    fence
    li a0, DST_MIN
    li a1, 2
    jal ra, verify_min
    jal ra, mark_pass

    vmax.vv v3, v1, v2
    li a0, DST_MAX
    vse32.v v3, (a0)
    fence
    li a0, DST_MAX
    li a1, 3
    jal ra, verify_max
    jal ra, mark_pass

    vsra.vi v3, v1, 1
    li a0, DST_SRA
    vse32.v v3, (a0)
    fence
    li a0, DST_SRA
    li a1, 4
    jal ra, verify_sra
    jal ra, mark_pass

    vmslt.vv v0, v1, v2
    vmerge.vvm v3, v1, v2, v0
    li a0, DST_MERGE
    vse32.v v3, (a0)
    fence
    li a0, DST_MERGE
    li a1, 5
    jal ra, verify_max
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

value_a:
    li t1, 3
    mul t1, t1, t0
    addi t1, t1, -24
    ret

value_b:
    li t2, 2
    mul t2, t2, t0
    addi t2, t2, -7
    ret

verify_mul:
    li t0, 0
1:
    lw t3, 0(a0)
    jal ra, value_a
    jal ra, value_b
    mul t4, t1, t2
    mv t1, t3
    mv t2, t4
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t5, VL
    blt t0, t5, 1b
    ret

verify_min:
    li t0, 0
1:
    lw t3, 0(a0)
    jal ra, value_a
    jal ra, value_b
    blt t1, t2, 2f
    mv t4, t2
    j 3f
2:
    mv t4, t1
3:
    mv t1, t3
    mv t2, t4
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t5, VL
    blt t0, t5, 1b
    ret

verify_max:
    li t0, 0
1:
    lw t3, 0(a0)
    jal ra, value_a
    jal ra, value_b
    blt t1, t2, 2f
    mv t4, t1
    j 3f
2:
    mv t4, t2
3:
    mv t1, t3
    mv t2, t4
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t5, VL
    blt t0, t5, 1b
    ret

verify_sra:
    li t0, 0
1:
    lw t3, 0(a0)
    jal ra, value_a
    srai t4, t1, 1
    mv t1, t3
    mv t2, t4
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t5, VL
    blt t0, t5, 1b
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
