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

    .equ SRC8,            0x10101000
    .equ DST8,            0x10101100
    .equ SRC16,           0x10101200
    .equ DST16,           0x10101300
    .equ SRC32,           0x10101400
    .equ DST32,           0x10101500
    .equ VL8,             32
    .equ VL16,            16
    .equ VL32,            16

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

    li t0, 1
    sw t0, 24(s0)

    li a0, SRC8
    li a1, DST8
    li t0, 0
1:
    slli t1, t0, 1
    addi t1, t1, 3
    sb t1, 0(a0)
    sb zero, 0(a1)
    addi a0, a0, 1
    addi a1, a1, 1
    addi t0, t0, 1
    li t2, VL8
    blt t0, t2, 1b

    li t0, VL8
    vsetvli t1, t0, e8, m1, ta, ma
    li a0, SRC8
    vle8.v v1, (a0)
    li a0, DST8
    vse8.v v1, (a0)
    fence
    li a0, DST8
    li a1, 1
    jal ra, verify_e8
    jal ra, mark_pass

    li a0, SRC16
    li a1, DST16
    li t0, 0
2:
    slli t1, t0, 2
    addi t1, t1, 0x123
    sh t1, 0(a0)
    sh zero, 0(a1)
    addi a0, a0, 2
    addi a1, a1, 2
    addi t0, t0, 1
    li t2, VL16
    blt t0, t2, 2b

    li t0, VL16
    vsetvli t1, t0, e16, m1, ta, ma
    li a0, SRC16
    vle16.v v1, (a0)
    li a0, DST16
    vse16.v v1, (a0)
    fence
    li a0, DST16
    li a1, 2
    jal ra, verify_e16
    jal ra, mark_pass

    li a0, SRC32
    li a1, DST32
    li t0, 0
3:
    li t1, 17
    mul t1, t1, t0
    addi t1, t1, -64
    sw t1, 0(a0)
    sw zero, 0(a1)
    addi a0, a0, 4
    addi a1, a1, 4
    addi t0, t0, 1
    li t2, VL32
    blt t0, t2, 3b

    li t0, VL32
    vsetvli t1, t0, e32, m1, ta, ma
    li a0, SRC32
    vle32.v v1, (a0)
    li a0, DST32
    vse32.v v1, (a0)
    fence
    li a0, DST32
    li a1, 3
    jal ra, verify_e32
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

verify_e8:
    li t0, 0
1:
    lbu t1, 0(a0)
    slli t2, t0, 1
    addi t2, t2, 3
    andi t2, t2, 0xff
    bne t1, t2, fail
    addi a0, a0, 1
    addi t0, t0, 1
    li t3, VL8
    blt t0, t3, 1b
    ret

verify_e16:
    li t0, 0
1:
    lhu t1, 0(a0)
    slli t2, t0, 2
    addi t2, t2, 0x123
    bne t1, t2, fail
    addi a0, a0, 2
    addi t0, t0, 1
    li t3, VL16
    blt t0, t3, 1b
    ret

verify_e32:
    li t0, 0
1:
    lw t1, 0(a0)
    li t2, 17
    mul t2, t2, t0
    addi t2, t2, -64
    bne t1, t2, fail
    addi a0, a0, 4
    addi t0, t0, 1
    li t3, VL32
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
