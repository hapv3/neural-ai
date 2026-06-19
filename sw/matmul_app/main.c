typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

#define REG_DMA_START (*(volatile uint32_t*)0x20000000)
#define REG_DMA_SRC   (*(volatile uint32_t*)0x20000020)
#define REG_DMA_DST   (*(volatile uint32_t*)0x20000040)
#define REG_DMA_LEN   (*(volatile uint32_t*)0x20000060)
#define REG_DMA_DONE  (*(volatile uint32_t*)0x20000080)

#define REG_SYS_W_PTR (*(volatile uint32_t*)0x20000100)
#define REG_SYS_I_PTR (*(volatile uint32_t*)0x20000104)
#define REG_SYS_O_PTR (*(volatile uint32_t*)0x20000108)
#define REG_SYS_DIM_M (*(volatile uint32_t*)0x2000010C)
#define REG_SYS_START (*(volatile uint32_t*)0x20000110)
#define REG_SYS_DONE  (*(volatile uint32_t*)0x20000114)

#define WEIGHT_PING_ADDR 0x10110000
#define IFM_PING_ADDR    0x10120000
#define OFM_PING_ADDR    0x10200000

#define EXT_MEM_WEIGHT   0x80000000
#define EXT_MEM_IFM      0x80001000
#define EXT_MEM_OFM      0x80002000

void dma_transfer(uint32_t src, uint32_t dst, uint32_t len) {
    REG_DMA_SRC = src;
    REG_DMA_DST = dst;
    REG_DMA_LEN = len;
    REG_DMA_DONE = 0; // Clear done flag
    REG_DMA_START = 1;

    while (REG_DMA_DONE == 0) {
        // Wait
    }
}

int main() {
    volatile uint32_t *start_flag = (volatile uint32_t*)0x10008004;
    volatile uint32_t *done_flag = (volatile uint32_t*)0x10008008;

    while (1) {
        // Wait for testbench to set start_flag to 1
        while (*start_flag == 0) {
            // wait
        }
        *start_flag = 0; // clear for next run

        uint32_t dim_m = *(volatile uint32_t*)0x10008000;
        if (dim_m == 0 || dim_m > 128) dim_m = 32; // Default safety check

        uint32_t weight_len = 32 * 32; // 1024 bytes (8-bit), fixed K=32
        uint32_t ifm_len = dim_m * 32; // 32 bytes per row * dim_m rows
        uint32_t ofm_len = dim_m * 32 * 4; // 128 bytes per row * dim_m rows

        // 1. DMA Weights L2 -> L1 (I-TCDM)
        dma_transfer(EXT_MEM_WEIGHT, WEIGHT_PING_ADDR, weight_len);

        // 2. DMA IFM L2 -> L1 (I-TCDM)
        dma_transfer(EXT_MEM_IFM, IFM_PING_ADDR, ifm_len);

        // 3. Configure and Start Systolic Array
        REG_SYS_W_PTR = WEIGHT_PING_ADDR;
        REG_SYS_I_PTR = IFM_PING_ADDR;
        REG_SYS_O_PTR = OFM_PING_ADDR;
        REG_SYS_DIM_M = dim_m;
        REG_SYS_DONE  = 0;
        REG_SYS_START = 1;

        // Wait for computation
        while (REG_SYS_DONE == 0) {
            // Wait
        }

        // 4. DMA OFM L1 (O-TCDM) -> L2
        dma_transfer(OFM_PING_ADDR, EXT_MEM_OFM, ofm_len);

        // Signal finish by setting a done flag in D-TCM
        *done_flag = 1;
    }

    return 0;
}
