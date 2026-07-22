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

    .status (NOLOAD) : {
        . = . + 0x100;
    } > D_TCM

    .data : {
        *(.rodata)
        *(.rodata.*)
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > D_TCM

    .bss (NOLOAD) : {
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
    } > D_TCM

    . = ALIGN(16);
    __stack_bottom = ORIGIN(D_TCM) + LENGTH(D_TCM) - 2K;
    __stack_top = ORIGIN(D_TCM) + LENGTH(D_TCM);

    ASSERT(SIZEOF(.text) <= LENGTH(I_TCM), "Neural-AI runtime exceeds 32 KiB ITCM")
    ASSERT(ADDR(.bss) + SIZEOF(.bss) <= __stack_bottom, "Neural-AI runtime exceeds DTCM before stack")
}
