// Gate only supported unmasked arithmetic. Keep arith_mask as a separate,
// currently failing diagnostic; do not relabel its excluded check as PASS.
#define SPATZ_MASK_TEST 0
#include "arith_mask.S"
