    .section .text
    .align 2

    .global spatz_vec_copy_i8
    .type spatz_vec_copy_i8, @function
spatz_vec_copy_i8:
    beqz a2, 2f
1:
    vsetvli t0, a2, e8, m1, ta, ma
    vle8.v v1, (a0)
    vse8.v v1, (a1)
    add a0, a0, t0
    add a1, a1, t0
    sub a2, a2, t0
    bnez a2, 1b
2:
    ret

    .global spatz_vec_zero_i8
    .type spatz_vec_zero_i8, @function
spatz_vec_zero_i8:
    beqz a1, 2f
1:
    vsetvli t0, a1, e8, m8, ta, ma
    vmv.v.i v8, 0
    vse8.v v8, (a0)
    add a0, a0, t0
    sub a1, a1, t0
    bnez a1, 1b
2:
    ret

    .global spatz_vec_zero_i32
    .type spatz_vec_zero_i32, @function
spatz_vec_zero_i32:
    beqz a1, 2f
1:
    vsetvli t0, a1, e32, m8, ta, ma
    vmv.v.i v8, 0
    vse32.v v8, (a0)
    slli t1, t0, 2
    add a0, a0, t1
    sub a1, a1, t0
    bnez a1, 1b
2:
    ret

    .global spatz_vec_strided_copy_i8
    .type spatz_vec_strided_copy_i8, @function
spatz_vec_strided_copy_i8:
    beqz a2, 2f
1:
    vsetvli t0, a2, e8, m8, ta, ma
    vlse8.v v8, (a0), a3
    vsse8.v v8, (a1), a4
    mul t1, t0, a3
    mul t2, t0, a4
    add a0, a0, t1
    add a1, a1, t2
    sub a2, a2, t0
    bnez a2, 1b
2:
    ret

    .global spatz_im2col_rect_i8
    .type spatz_im2col_rect_i8, @function
spatz_im2col_rect_i8:
    beqz a2, 4f
    beqz a3, 4f
1:
    mv t3, a0
    mv t4, a1
    mv t5, a3
2:
    vsetvli t0, t5, e8, m8, ta, ma
    vlse8.v v8, (t3), a4
    vsse8.v v8, (t4), a5
    mul t1, t0, a4
    mul t2, t0, a5
    add t3, t3, t1
    add t4, t4, t2
    sub t5, t5, t0
    bnez t5, 2b
    add a0, a0, a6
    add a1, a1, a7
    addi a2, a2, -1
    bnez a2, 1b
4:
    ret

    .global spatz_im2col_rect_channels_i8
    .type spatz_im2col_rect_channels_i8, @function
spatz_im2col_rect_channels_i8:
    beqz a2, 5f
    beqz a3, 5f
    beqz a4, 5f
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s3, 8(sp)
    sw s4, 12(sp)
    sw s5, 16(sp)
    mv s0, a3
    mv s1, a4
    lw s3, 32(sp)
1:
    mv t3, a0
    mv t4, a1
    mv t5, s0
2:
    mv t0, t3
    mv t1, t4
    mv t2, s1
3:
    vsetvli t6, t2, e8, m8, ta, ma
    vlse8.v v8, (t0), a5
    vsse8.v v8, (t1), a6
    mul s4, t6, a5
    mul s5, t6, a6
    add t0, t0, s4
    add t1, t1, s5
    sub t2, t2, t6
    bnez t2, 3b
    add t3, t3, a7
    add t4, t4, s3
    addi t5, t5, -1
    bnez t5, 2b
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, 1b
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s3, 8(sp)
    lw s4, 12(sp)
    lw s5, 16(sp)
    addi sp, sp, 32
5:
    ret

    .global spatz_vec_relu_i8
    .type spatz_vec_relu_i8, @function
spatz_vec_relu_i8:
    beqz a1, 2f
    li t1, 0
1:
    vsetvli t0, a1, e8, m1, ta, ma
    vle8.v v1, (a0)
    vmax.vx v1, v1, t1
    vse8.v v1, (a0)
    add a0, a0, t0
    sub a1, a1, t0
    bnez a1, 1b
2:
    ret

    .global spatz_requant_i32_to_i32
    .type spatz_requant_i32_to_i32, @function
spatz_requant_i32_to_i32:
    beqz a2, 3f
1:
    vsetvli t0, a2, e32, m1, ta, ma
    vle32.v v1, (a0)
    vmul.vx v1, v1, a3
    beqz a4, 2f
    vsra.vx v1, v1, a4
2:
    vmax.vx v1, v1, a5
    vmin.vx v1, v1, a6
    vse32.v v1, (a1)
    slli t1, t0, 2
    add a0, a0, t1
    add a1, a1, t1
    sub a2, a2, t0
    bnez a2, 1b
3:
    ret

    .global spatz_mul_i8_q7
    .type spatz_mul_i8_q7, @function
spatz_mul_i8_q7:
    beqz a3, 3f
    li t1, 0xff
    li t2, 0xff00
    li t3, 7
1:
    vsetvli t0, a3, e8, m4, ta, ma
    vle8.v v8, (a0)
    vle8.v v16, (a1)
    vwmul.vv v24, v8, v16
    vsetvli zero, t0, e16, m8, ta, ma
    vsra.vx v24, v24, t3
    li t4, -128
    li t5, 127
    vmax.vx v24, v24, t4
    vmin.vx v24, v24, t5
    lui t6, 0x10140
    vse16.v v24, (t6)
    mv a4, t6
    mv a5, a2
    srli a6, t0, 2
    beqz a6, 4f
2:
    lw a7, 0(a4)
    lw t6, 4(a4)
    and t4, a7, t1
    srli a7, a7, 8
    and a7, a7, t2
    or t4, t4, a7
    and a7, t6, t1
    srli t6, t6, 8
    and t6, t6, t2
    or a7, a7, t6
    slli a7, a7, 16
    or t4, t4, a7
    sw t4, 0(a5)
    addi a4, a4, 8
    addi a5, a5, 4
    addi a6, a6, -1
    bnez a6, 2b
4:
    andi a6, t0, 3
    beqz a6, 5f
6:
    lh a7, 0(a4)
    sb a7, 0(a5)
    addi a4, a4, 2
    addi a5, a5, 1
    addi a6, a6, -1
    bnez a6, 6b
5:
    add a0, a0, t0
    add a1, a1, t0
    add a2, a2, t0
    sub a3, a3, t0
    bnez a3, 1b
3:
    ret

    .global spatz_maxpool2d_5x5s1p2_c32_i8
    .type spatz_maxpool2d_5x5s1p2_c32_i8, @function
spatz_maxpool2d_5x5s1p2_c32_i8:
    beqz a2, 9f
    beqz a3, 9f
    addi sp, sp, -48
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)
    sw s7, 28(sp)
    sw s8, 32(sp)
    sw s9, 36(sp)
    sw s10, 40(sp)
    sw s11, 44(sp)

    mv s0, a0
    mv s1, a1
    mv s2, a2
    mv s3, a3
    slli s4, s3, 5
    li t0, 32
    vsetvli zero, t0, e8, m1, ta, ma
    lui s11, 0x10140

    mv s10, s11
    li s5, 0
1:
    bgeu s5, s2, 8f
    li s6, 0
2:
    bgeu s6, s3, 7f

    li t0, -128
    vmv.v.x v8, t0

    li t0, 2
    bgtu s6, t0, 3f
    li s7, 0
    j 4f
3:
    addi s7, s6, -2
4:
    addi s8, s6, 3
    bleu s8, s3, 5f
    mv s8, s3
5:
    mul t2, s5, s4
    slli t3, s7, 5
    add t2, t2, t3
    add t2, s0, t2
    mv t4, s7
6:
    bgeu t4, s8, 15f
    vle8.v v9, (t2)
    vmax.vv v8, v8, v9
    addi t2, t2, 32
    addi t4, t4, 1
    j 6b
15:
    vse8.v v8, (s10)
    addi s10, s10, 32
    addi s6, s6, 1
    j 2b
7:
    addi s5, s5, 1
    j 1b
8:
    li s5, 0
10:
    bgeu s5, s2, 18f
    li s6, 0
11:
    bgeu s6, s3, 17f

    li t0, -128
    vmv.v.x v8, t0

    li t0, 2
    bgtu s5, t0, 12f
    li s7, 0
    j 13f
12:
    addi s7, s5, -2
13:
    addi s8, s5, 3
    bleu s8, s2, 14f
    mv s8, s2
14:
    mul t2, s7, s4
    slli t3, s6, 5
    add t2, t2, t3
    add t2, s11, t2
    mv t4, s7
16:
    bgeu t4, s8, 19f
    vle8.v v9, (t2)
    vmax.vv v8, v8, v9
    add t2, t2, s4
    addi t4, t4, 1
    j 16b
19:
    vse8.v v8, (s1)
    addi s1, s1, 32
    addi s6, s6, 1
    j 11b
17:
    addi s5, s5, 1
    j 10b
18:
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    lw s8, 32(sp)
    lw s9, 36(sp)
    lw s10, 40(sp)
    lw s11, 44(sp)
    addi sp, sp, 48
9:
    ret
