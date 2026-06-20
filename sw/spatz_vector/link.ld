OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    I_TCM (rx)  : ORIGIN = 0x10000000, LENGTH = 32K
    D_TCM (rw)  : ORIGIN = 0x10008000, LENGTH = 32K
}

SECTIONS
{
    .text : {
        *(.text.init)
        *(.text)
        *(.text.*)
    } > I_TCM

    .rodata : {
        *(.rodata)
        *(.rodata.*)
    } > D_TCM

    .data : {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > D_TCM

    .bss : {
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
    } > D_TCM

    . = ALIGN(8);
    __stack_top = ORIGIN(D_TCM) + LENGTH(D_TCM);
}
