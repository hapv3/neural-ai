OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    itcm (rx)  : ORIGIN = 0x10000000, LENGTH = 32K
    dtcm (rw)  : ORIGIN = 0x10008000, LENGTH = 8K
}

SECTIONS
{
    .text :
    {
        *(.text.init)
        *(.text)
        *(.text.*)
    } > itcm

    .status (NOLOAD) : {
        . = . + 0x100;
    } > dtcm

    .rodata :
    {
        *(.rodata)
        *(.rodata.*)
    } > dtcm

    .data :
    {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > dtcm

    .bss :
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        __bss_end = .;
    } > dtcm

    /* Define Stack top at the end of D-TCM */
    __stack_top = ORIGIN(dtcm) + LENGTH(dtcm);
}
