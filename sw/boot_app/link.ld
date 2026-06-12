OUTPUT_ARCH(riscv)
ENTRY(main)

MEMORY
{
    TCDM (rwx) : ORIGIN = 0x10000000, LENGTH = 512K
}

SECTIONS
{
    .text : {
        *(.text.init)
        *(.text)
        *(.text.*)
    } > TCDM

    .rodata : {
        *(.rodata)
        *(.rodata.*)
    } > TCDM

    .data : {
        *(.data)
        *(.data.*)
        *(.sdata)
    } > TCDM

    .bss : {
        *(.bss)
        *(.bss.*)
        *(.sbss)
    } > TCDM

    /* Define a stack pointer */
    PROVIDE(__stack_top = ORIGIN(TCDM) + LENGTH(TCDM) - 8);
}
