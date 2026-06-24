package npu_cluster_pkg;

    //---------------------------------------------------------
    // System Definitions
    //---------------------------------------------------------
    localparam int unsigned AXI_ADDR_WIDTH = 64;
    localparam int unsigned AXI_DATA_WIDTH = 256; // 32 Bytes
    localparam int unsigned AXI_HOST_DATA_WIDTH = 32; // Host AXI-Lite boot port
    localparam int unsigned AXI_USER_WIDTH = 1;
    localparam int unsigned AXI_ID_WIDTH   = 8;

    localparam int unsigned OBI_ADDR_WIDTH = 32;
    localparam int unsigned OBI_DATA_WIDTH = 256;
    localparam int unsigned OBI_STRB_WIDTH = OBI_DATA_WIDTH / 8;
    localparam int unsigned ITCM_DATA_WIDTH = 32;
    localparam int unsigned SNITCH_D_DATA_WIDTH = 32;
    localparam int unsigned DTCM_DATA_WIDTH = 32;
    localparam int unsigned MMIO_DATA_WIDTH = 32;

    //---------------------------------------------------------
    // TCDM Configuration (512KB Total)
    //---------------------------------------------------------
    localparam int unsigned TCDM_BANK_SIZE = 32 * 1024; // 32 KB
    localparam int unsigned I_TCDM_NUM_BANKS = 12; // 384 KB
    localparam int unsigned O_TCDM_NUM_BANKS = 4;  // 128 KB
    localparam int unsigned TCDM_NUM_BANKS = I_TCDM_NUM_BANKS + O_TCDM_NUM_BANKS;
    localparam int unsigned TCDM_TOTAL_SIZE = TCDM_NUM_BANKS * TCDM_BANK_SIZE;

    //---------------------------------------------------------
    // Memory Map (Base Addresses)
    //---------------------------------------------------------
    // I-TCDM (Instruction memory for Snitch) - 32KB
    localparam logic [31:0] ITCDM_BASE_ADDR = 32'h1000_0000;
    localparam logic [31:0] ITCDM_SIZE      = 32'h0000_8000;

    // D-TCM (Snitch private scalar data) - 32KB
    localparam logic [31:0] DTCM_BASE_ADDR  = 32'h1000_8000;
    localparam logic [31:0] DTCM_SIZE       = 32'h0000_8000;

    // D-TCDM (Shared Vector Data: Weights, IFM) - 384KB
    localparam logic [31:0] I_TCDM_BASE_ADDR = 32'h1010_0000;
    localparam logic [31:0] I_TCDM_SIZE      = I_TCDM_NUM_BANKS * TCDM_BANK_SIZE;

    // O-TCDM (Shared Vector Data: OFM) - 128KB
    localparam logic [31:0] O_TCDM_BASE_ADDR = 32'h1020_0000;
    localparam logic [31:0] O_TCDM_SIZE      = O_TCDM_NUM_BANKS * TCDM_BANK_SIZE;

    // Logical Buffers Pointers inside I-TCDM
    localparam logic [31:0] WEIGHT_PING_ADDR = 32'h1011_0000;
    localparam logic [31:0] WEIGHT_PONG_ADDR = 32'h1012_0000;
    localparam logic [31:0] IFM_PING_ADDR    = 32'h1012_0000;
    localparam logic [31:0] IFM_PONG_ADDR    = 32'h1014_C800;
    
    // Logical Buffers Pointers inside O-TCDM
    localparam logic [31:0] OFM_PING_ADDR    = 32'h1020_0000;
    localparam logic [31:0] OFM_PONG_ADDR    = 32'h1020_C800;

    // Memory-Mapped Registers (MMR) for Systolic Array
    localparam logic [31:0] SYSTOLIC_MMR_BASE = 32'h2000_0000;
    localparam logic [31:0] SYSTOLIC_MMR_SIZE = 32'h0000_0100; // 256 Bytes

    // Register Offsets
    localparam logic [31:0] REG_SYSTOLIC_STATUS = 32'h00; // RO
    localparam logic [31:0] REG_SYSTOLIC_CTRL   = 32'h04; // RW: bit0=clear_acc, bit1=weight_load, bit2=compute_en

    //---------------------------------------------------------
    // OBI Interface Macros / Structs
    // (Used to define standardized OBI requests/responses)
    //---------------------------------------------------------
    typedef struct packed {
        logic                      req;
        logic                      we;
        logic [OBI_STRB_WIDTH-1:0] be;
        logic [OBI_ADDR_WIDTH-1:0] addr;
        logic [OBI_DATA_WIDTH-1:0] wdata;
    } obi_req_t;

    typedef struct packed {
        logic                      gnt;
        logic                      rvalid;
        logic [OBI_DATA_WIDTH-1:0] rdata;
    } obi_rsp_t;

endpackage
