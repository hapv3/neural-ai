    .section .text.init
    .global _start

_start:
    /* Set up stack pointer */
    la sp, __stack_top

    /* Call main */
    call main

    /* Loop forever if main returns */
1:  j 1b
