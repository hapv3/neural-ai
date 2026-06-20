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
