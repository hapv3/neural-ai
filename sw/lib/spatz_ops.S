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

    .global spatz_pack_dfl16_chw_tile_i8
    .type spatz_pack_dfl16_chw_tile_i8, @function
spatz_pack_dfl16_chw_tile_i8:
    beqz a2, 2f
    li a4, 16
    vsetvli t0, a4, e8, m8, ta, ma
1:
    vlse8.v v8, (a0), a3
    vse8.v v8, (a1)
    addi a0, a0, 1
    addi a1, a1, 32
    addi a2, a2, -1
    bnez a2, 1b
2:
    ret

    .global spatz_pack_dfl16_c32_tile_i8
    .type spatz_pack_dfl16_c32_tile_i8, @function
spatz_pack_dfl16_c32_tile_i8:
    beqz a2, 2f
    li a4, 16
    vsetvli t0, a4, e8, m8, ta, ma
1:
    vle8.v v8, (a0)
    vse8.v v8, (a1)
    add a0, a0, a3
    addi a1, a1, 32
    addi a2, a2, -1
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

    /*
     * Apply Regor/TFLite's signed fixed-point scale using an e32 vector pair
     * for the low and high halves of each exact 64-bit product.  Spatz is an
     * ELEN=32 implementation, so this deliberately avoids e64 operations.
     * v8 and v16 are scratch groups; source and result are e32,m4 groups.
     */
    .macro spatz_scale_i32 source, result, multiplier, shift, double_shift
    vmul.vx v8, \source, \multiplier
    vmulh.vx v16, \source, \multiplier
    beqz \shift, .Lscale_no_shift\@

    vmv.v.i v24, 0
    vmv.v.i v28, 0
    li t1, 32
    bgtu \shift, t1, .Lscale_round_high\@
    li t2, 1
    addi t3, \shift, -1
    sll t2, t2, t3
    vmv.v.x v24, t2
    j .Lscale_double_round\@
.Lscale_round_high\@:
    li t2, 1
    addi t3, \shift, -33
    sll t2, t2, t3
    vmv.v.x v28, t2

.Lscale_double_round\@:
    li t2, 31
    sub t2, t2, \double_shift
    bleu \shift, t2, .Lscale_add_round\@
    li t2, 1
    li t3, 30
    sub t3, t3, \double_shift
    sll t2, t2, t3
    vsra.vi v28, \source, 31
    li t3, 32
    bgtu \shift, t3, .Lscale_double_high\@
    slli t2, t2, 1
    vmul.vx v28, v28, t2
    srli t2, t2, 1
    vadd.vx v28, v28, t2
    vadd.vv v24, v24, v28
    vmv.v.i v28, 0
    j .Lscale_add_round\@
.Lscale_double_high\@:
    vadd.vv v28, v28, v28
    vmul.vx v24, v28, t2
    vadd.vx v24, v24, t2
    vsra.vi v28, \source, 31
    li t3, 1
    addi t1, \shift, -33
    sll t3, t3, t1
    vadd.vx v28, v28, t3

.Lscale_add_round\@:
    vadd.vv v16, v16, v28
    vadd.vv \result, v8, v24
    vor.vv v28, v8, v24
    vand.vv v8, v8, v24
    vxor.vi v24, \result, -1
    vand.vv v28, v28, v24
    vor.vv v8, v8, v28
    vsrl.vi v8, v8, 31
    vadd.vv v16, v16, v8

    li t1, 32
    beq \shift, t1, .Lscale_shift_32\@
    bgtu \shift, t1, .Lscale_shift_high\@
    li t2, 32
    sub t2, t2, \shift
    vsrl.vx v8, \result, \shift
    vsll.vx v24, v16, t2
    vor.vv \result, v8, v24
    j .Lscale_done\@
.Lscale_shift_32\@:
    vadd.vx \result, v16, zero
    j .Lscale_done\@
.Lscale_shift_high\@:
    addi t2, \shift, -32
    vsra.vx \result, v16, t2
    j .Lscale_done\@
.Lscale_no_shift\@:
    vadd.vx \result, v8, zero
.Lscale_done\@:
    .endm

    .global spatz_dfl_q8_to_i8
    .type spatz_dfl_q8_to_i8, @function
spatz_dfl_q8_to_i8:
    beqz a3, .Ldfl_q8_done
    li t0, 17
    bltu a5, t0, .Ldfl_q8_wide
    mv t5, a0
    mv t6, a3
    addi t1, a5, -16
    addi t2, t1, -1
    li t3, 1
    sll t3, t3, t2
.Ldfl_q8_loop:
    vsetvli t0, a3, e16, m4, ta, ma
    vle16.v v8, (a0)
    vmulhu.vx v8, v8, a4
    vadd.vx v8, v8, t3
    vsrl.vx v8, v8, t1
    vadd.vx v8, v8, a6
    vmax.vx v8, v8, a7
    lw t4, 0(sp)
    vmin.vx v8, v8, t4
    vse16.v v8, (a0)
    slli t2, t0, 1
    add a0, a0, t2
    sub a3, a3, t0
    bnez a3, .Ldfl_q8_loop
    li t2, 2
.Ldfl_q8_pack:
    vsetvli t0, t6, e8, m8, ta, ma
    vlse8.v v8, (t5), t2
    vse8.v v8, (a1)
    slli t4, t0, 1
    add t5, t5, t4
    add a1, a1, t0
    sub t6, t6, t0
    bnez t6, .Ldfl_q8_pack
    ret

.Ldfl_q8_wide:
    mv t4, a2
    mv t5, a2
    mv t6, a3
    beqz a5, .Ldfl_q8_wide_loop
    li t1, 1
    addi t2, a5, -1
    sll t1, t1, t2
.Ldfl_q8_wide_loop:
    vsetvli t0, a3, e16, m4, ta, ma
    vle16.v v24, (a0)
    vwmulu.vx v8, v24, a4
    vsetvli zero, t0, e32, m8, ta, ma
    beqz a5, .Ldfl_q8_wide_unshifted
    vadd.vx v8, v8, t1
    vsrl.vx v8, v8, a5
.Ldfl_q8_wide_unshifted:
    vadd.vx v8, v8, a6
    vmax.vx v8, v8, a7
    lw t2, 0(sp)
    vmin.vx v8, v8, t2
    vse32.v v8, (t5)
    slli t2, t0, 1
    add a0, a0, t2
    slli t2, t0, 2
    add t5, t5, t2
    sub a3, a3, t0
    bnez a3, .Ldfl_q8_wide_loop
    li t2, 4
.Ldfl_q8_wide_pack:
    vsetvli t0, t6, e8, m8, ta, ma
    vlse8.v v8, (t4), t2
    vse8.v v8, (a1)
    slli t3, t0, 2
    add t4, t4, t3
    add a1, a1, t0
    sub t6, t6, t0
    bnez t6, .Ldfl_q8_wide_pack
.Ldfl_q8_done:
    ret

    .global spatz_quantized_add_i8
    .type spatz_quantized_add_i8, @function
spatz_quantized_add_i8:
    beqz a3, .Lquant_add_done
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
    mv s4, a4
    lw s6, 0(s4)
    lw s7, 4(s4)
    lw s8, 8(s4)
    lw s9, 12(s4)
    lw s10, 16(s4)
    lw s11, 20(s4)

.Lquant_add_loop:
    vsetvli s5, s3, e8, m1, ta, mu
    vle8.v v1, (s0)
    vle8.v v2, (s1)
    li t1, 1
    vwmul.vx v8, v1, t1
    vwmul.vx v10, v2, t1
    vsetvli zero, s5, e16, m2, ta, mu
    vwmul.vx v12, v8, t1
    vwmul.vx v20, v10, t1
    vsetvli zero, s5, e32, m4, ta, mu
    lw t1, 24(s4)
    lw t2, 28(s4)
    vsub.vx v12, v12, t1
    vsub.vx v20, v20, t2

    lw t4, 44(s4)
    spatz_scale_i32 v12, v12, s6, s7, t4
    spatz_scale_i32 v20, v20, s8, s9, t4
    vadd.vv v12, v12, v20
    li t4, 0
    spatz_scale_i32 v12, v12, s10, s11, t4

    lw t1, 32(s4)
    lw t2, 36(s4)
    lw t3, 40(s4)
    vadd.vx v12, v12, t1
    vmax.vx v12, v12, t2
    vmin.vx v12, v12, t3
    lui t1, 0x10140
    vse32.v v12, (t1)
    mv t2, s2
    mv t3, s5
.Lquant_add_pack:
    lw t4, 0(t1)
    sb t4, 0(t2)
    addi t1, t1, 4
    addi t2, t2, 1
    addi t3, t3, -1
    bnez t3, .Lquant_add_pack

    add s0, s0, s5
    add s1, s1, s5
    add s2, s2, s5
    sub s3, s3, s5
    bnez s3, .Lquant_add_loop

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
.Lquant_add_done:
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

    .global spatz_upsample_nearest2x_c32_i8
    .type spatz_upsample_nearest2x_c32_i8, @function
spatz_upsample_nearest2x_c32_i8:
    beqz a2, 5f
    beqz a3, 5f
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)

    mv s0, a0
    mv s1, a1
    mv s2, a2
    mv s3, a3
    slli s4, s3, 5
    slli s5, s3, 6
    li t0, 32
    vsetvli zero, t0, e8, m1, ta, ma
1:
    beqz s2, 4f
    mv t0, s0
    mv t1, s1
    add t2, s1, s5
    mv s6, s3
2:
    beqz s6, 3f
    vle8.v v8, (t0)
    vse8.v v8, (t1)
    addi t3, t1, 32
    vse8.v v8, (t3)
    vse8.v v8, (t2)
    addi t3, t2, 32
    vse8.v v8, (t3)
    addi t0, t0, 32
    addi t1, t1, 64
    addi t2, t2, 64
    addi s6, s6, -1
    j 2b
3:
    add s0, s0, s4
    slli t0, s5, 1
    add s1, s1, t0
    addi s2, s2, -1
    j 1b
4:
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    addi sp, sp, 32
5:
    ret
