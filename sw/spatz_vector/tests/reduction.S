    .option norvc

    .equ SIG_BASE,        0x10008000
    .equ SIG_STATUS,      0x10008000
    .equ SIG_PASS_COUNT,  0x10008004
    .equ SIG_FAIL_TEST,   0x10008008
    .equ SIG_FAIL_INDEX,  0x1000800c
    .equ SIG_FAIL_GOT,    0x10008010
    .equ SIG_FAIL_EXP,    0x10008014

    .equ SRC,             0x10103000
    .equ DST_SUM,         0x10103100
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

    li a0, SRC
    li t0, 0
1:
    addi t1, t0, 1
    sw t1, 0(a0)
    addi a0, a0, 4
    addi t0, t0, 1
    li t2, VL
    blt t0, t2, 1b

    li t0, VL
    vsetvli t1, t0, e32, m1, ta, ma
    li a0, SRC
    vle32.v v1, (a0)
    vmv.v.i v2, 0
    vredsum.vs v3, v1, v2
    li a0, DST_SUM
    vse32.v v3, (a0)
    fence

    lw t1, 0(a0)
    li t2, 136
    li t0, 0
    li a1, 1
    bne t1, t2, fail
    li t0, SIG_PASS_COUNT
    li t1, 1
    sw t1, 0(t0)
    li t0, 0xdeadbeef
    li t1, SIG_STATUS
    sw t0, 0(t1)
done:
    j done

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
