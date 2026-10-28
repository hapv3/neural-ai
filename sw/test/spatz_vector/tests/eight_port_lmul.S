    .option norvc
    .section .text.init
    .global _start
_start:
    la sp, __stack_top
    li s0, 0x10008000
    sw zero, 0(s0)
    # VID is not implemented in this configuration. Seed indices with scalar
    # stores, then exercise the supported eight-port vector load path.
    li a0, 0x10106000
    li t0, 0
seed:
    sb t0, 0(a0)
    addi a0, a0, 1
    addi t0, t0, 1
    li t1, 512
    blt t0, t1, seed
    # Exercise full and partial LMUL=8 native-port transfers without relying
    # on the pinned core's unreliable masked destination writeback path.
    li t0, 512
    vsetvli t1, t0, e8, m8, tu, mu
    bne t0, t1, fail
    li a0, 0x10106000
    vle8.v v8, (a0)
    vadd.vi v24, v8, 7
    li a0, 0x10107000
    vse8.v v24, (a0)
    fence
    # Check the full store before the tail overwrite can hide any bad bytes.
    li t0, 0
check_full:
    addi t2, t0, 7
    andi t2, t2, 255
    lbu t3, 0(a0)
    bne t2, t3, fail
    addi a0, a0, 1
    addi t0, t0, 1
    li t1, 512
    blt t0, t1, check_full
    li t0, 1
    sw t0, 4(s0)
    # Update 509 bytes; the last three bytes must retain the full-store values.
    li t0, 509
    vsetvli t1, t0, e8, m8, tu, mu
    bne t0, t1, fail
    vadd.vi v24, v8, 9
    li a0, 0x10107000
    vse8.v v24, (a0)
    fence
    li a0, 0x10107000
    li t0, 0
check:
    li t1, 509
    blt t0, t1, updated
    addi t2, t0, 7
    j expected
updated:
    addi t2, t0, 9
expected:
    andi t2, t2, 255
    lbu t3, 0(a0)
    bne t2, t3, fail
    addi a0, a0, 1
    addi t0, t0, 1
    li t1, 512
    blt t0, t1, check
    li t0, 2
    sw t0, 4(s0)
    li t0, 0xdeadbeef
    sw t0, 0(s0)
    li t0, 0x20002018
    li t1, 1
    sw t1, 0(t0)
done:
    j done
fail:
    sw t0, 12(s0)
    sw t3, 16(s0)
    sw t2, 20(s0)
    li t0, 0xbad00001
    sw t0, 0(s0)
    j done
