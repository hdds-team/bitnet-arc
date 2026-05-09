/*
 * bench/test_kv4_packing.cpp -- unit tests for kv4 VNNI INT2 packing.
 *
 * Per @naskel design v4 review fold #2: "deux petites fonctions
 * pures très testables ... Avec des tests unitaires sur fixture".
 *
 * Tests the host-callable helpers in src/kernel_v4_packing.h before
 * they are embedded into the ESIMD kernel. The probe
 * bench/probe_esimd_int2.cpp used uniform 0x55555555 = constant
 * matrix invariant to all permutations; this test uses
 * heterogeneous fixtures specifically designed to exercise the
 * layout.
 *
 * Build: cpp host-only, no SYCL, no OpenCL. g++ standard.
 */

#include "../src/kernel_v4_packing.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bitnet_arc::kv4;

namespace {
int g_pass = 0, g_fail = 0;
}

#define EXPECT_EQ(a, b) do {                                            \
    if ((a) != (b)) {                                                   \
        std::fprintf(stderr, "FAIL %s:%d  %s != %s   (0x%x != 0x%x)\n", \
                     __FILE__, __LINE__, #a, #b,                        \
                     (unsigned)(a), (unsigned)(b));                     \
        ++g_fail;                                                       \
    } else {                                                            \
        ++g_pass;                                                       \
    }                                                                   \
} while (0)

/* --- ternary <-> int2 signed encoding -------------------------------- */

void test_ternary_to_int2_signed_roundtrip() {
    EXPECT_EQ(ternary_to_int2_signed(-1), 0x3u);
    EXPECT_EQ(ternary_to_int2_signed( 0), 0x0u);
    EXPECT_EQ(ternary_to_int2_signed(+1), 0x1u);
    EXPECT_EQ(int2_signed_to_ternary(0x0u), 0);
    EXPECT_EQ(int2_signed_to_ternary(0x1u), 1);
    EXPECT_EQ(int2_signed_to_ternary(0x3u), -1);
}

/* --- pack_b_fragment_vnni_int2 invariants --------------------------- */

void test_pack_b_all_zero() {
    /* All-zero ternary input -> all-zero VNNI dwords. */
    std::int8_t ternary[64 * 16] = {};
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary, out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0u);
}

void test_pack_b_all_plus_one() {
    /* All +1 ternary -> each dword = 0x55555555 (16 reps of 01). */
    std::vector<std::int8_t> ternary(64 * 16, 1);
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary.data(), out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0x55555555u);
}

void test_pack_b_all_minus_one() {
    /* All -1 ternary -> each dword = 0xFFFFFFFF (16 reps of 11). */
    std::vector<std::int8_t> ternary(64 * 16, -1);
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary.data(), out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0xFFFFFFFFu);
}

void test_pack_b_single_element() {
    /* Set ternary[k=5, n=3] = +1, all others = 0. */
    std::vector<std::int8_t> ternary(64 * 16, 0);
    ternary[5 * 16 + 3] = 1;
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary.data(), out);
    /* k=5 is in dword d=0 (k<16), at bit pos (5%16)*2 = 10.
     * Column n=3 is at out[d*16 + n] = out[3]. */
    EXPECT_EQ(out[3], (0x1u << 10));
    /* All other 63 dwords must be zero. */
    for (unsigned i = 0; i < 64; ++i) {
        if (i != 3) EXPECT_EQ(out[i], 0u);
    }
}

void test_pack_b_dword_boundary() {
    /* Set ternary[k=16, n=7] = -1 (first element of dword d=1). */
    std::vector<std::int8_t> ternary(64 * 16, 0);
    ternary[16 * 16 + 7] = -1;
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary.data(), out);
    /* k=16 -> d=1, bit_pos = (16%16)*2 = 0. Column n=7 -> out[1*16+7]=out[23]. */
    EXPECT_EQ(out[23], 0x3u);  /* low 2 bits set to 11 */
    for (unsigned i = 0; i < 64; ++i) {
        if (i != 23) EXPECT_EQ(out[i], 0u);
    }
}

void test_pack_b_last_dword() {
    /* k=63 (last in fragment), n=0. */
    std::vector<std::int8_t> ternary(64 * 16, 0);
    ternary[63 * 16 + 0] = 1;
    std::uint32_t out[64];
    pack_b_fragment_vnni_int2(ternary.data(), out);
    /* k=63 -> d=3, bit_pos = (63%16)*2 = 30. n=0 -> out[3*16+0]=out[48]. */
    EXPECT_EQ(out[48], (0x1u << 30));
    for (unsigned i = 0; i < 64; ++i) {
        if (i != 48) EXPECT_EQ(out[i], 0u);
    }
}

/* --- pack_a_fragment_vnni_int2 ------------------------------------- */

/* Helper: encode a host float as FP16 bits via the oracle helper. */
inline std::uint16_t f32_to_h(float f) {
    return bitnet_arc_fp32_to_fp16(f);
}

void test_pack_a_all_zero() {
    std::uint16_t a[8 * 64] = {};
    float s_a[8];
    for (unsigned m = 0; m < 8; ++m) s_a[m] = 1.0f;
    std::uint32_t out[32];
    pack_a_fragment_vnni_int2(a, s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0u);
}

void test_pack_a_all_plus_one_unit_scale() {
    /* a = 1.0 everywhere, s_a = 1.0 -> q = 1 -> 0b01.
     * Each dword should be 0x55555555. */
    std::vector<std::uint16_t> a(8 * 64, f32_to_h(1.0f));
    float s_a[8];
    for (unsigned m = 0; m < 8; ++m) s_a[m] = 1.0f;
    std::uint32_t out[32];
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0x55555555u);
}

void test_pack_a_quant_clamp() {
    /* a = 0.4 with s_a = 1.0 -> round to 0. */
    std::vector<std::uint16_t> a(8 * 64, f32_to_h(0.4f));
    float s_a[8];
    for (unsigned m = 0; m < 8; ++m) s_a[m] = 1.0f;
    std::uint32_t out[32];
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0u);

    /* a = 0.6 -> round to 1. */
    for (auto& v : a) v = f32_to_h(0.6f);
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0x55555555u);

    /* a = 5.0 with s_a = 1.0 -> round 5, clamped to 1. */
    for (auto& v : a) v = f32_to_h(5.0f);
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0x55555555u);

    /* a = -5.0 -> -1, clamped. */
    for (auto& v : a) v = f32_to_h(-5.0f);
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], 0xFFFFFFFFu);
}

void test_pack_a_per_row_scale() {
    /* a is a different value per row, s_a sized so q ≈ +1. */
    std::vector<std::uint16_t> a(8 * 64);
    float s_a[8];
    for (unsigned m = 0; m < 8; ++m) {
        const float v = 1.5f + 0.1f * m;
        for (unsigned k = 0; k < 64; ++k) a[m * 64 + k] = f32_to_h(v);
        s_a[m] = v;  /* so a/s_a = 1 -> q = 1 */
    }
    std::uint32_t out[32];
    pack_a_fragment_vnni_int2(a.data(), s_a, out);
    /* For row m, q=1, all dwords for that m = 0x55555555.
     * Layout: out[d*8 + m]. */
    for (unsigned d = 0; d < 4; ++d) {
        for (unsigned m = 0; m < 8; ++m) {
            EXPECT_EQ(out[d * 8 + m], 0x55555555u);
        }
    }
}

/* --- TQ2_0 unpack ---------------------------------------------------- */

void test_tq2_0_unpack_all_codes() {
    /* Build a TQ2_0 block with all-zero qs (= all -1 ternary). */
    bitnet_arc_tq2_0_block blk = {};
    blk.d = f32_to_h(1.0f);
    /* qs all zero -> code = 0 -> ternary -1. */
    std::int8_t out[256];
    unpack_tq2_0_block_to_ternary(blk, out);
    for (unsigned k = 0; k < 256; ++k) EXPECT_EQ(out[k], -1);

    /* qs all 0xFF -> every code = 0b11 = 3 (unused, but unpacks to +1
     * in current mapping since (code==0?-1:(code==1?0:1)) maps 3 to 1).
     * This is acceptable -- valid TQ2_0 should never have code 3. */
    std::memset(blk.qs, 0xFF, sizeof(blk.qs));
    unpack_tq2_0_block_to_ternary(blk, out);
    for (unsigned k = 0; k < 256; ++k) EXPECT_EQ(out[k], 1);

    /* qs pattern producing alternating 0/1 codes via formula. */
    std::memset(blk.qs, 0, sizeof(blk.qs));
    /* k=0: byte=0, shift=0, code = blk.qs[0] & 3 = ?
     * Set blk.qs[0] = 0b01010101 = 0x55. Codes for k=0,1,2,3 (same byte,
     * shifts 0,0,0,0) -> all = 1 -> ternary 0.
     * Hmm, formula: byte = (k>>7)*32+(k&31), shift = ((k>>5)&3)*2.
     * For k=0: byte=0, shift=0, code = qs[0]&0x3 = 1 -> ternary 0.
     * For k=1: byte=1, shift=0. */
    blk.qs[0] = 0x01;  /* code=1 at low 2 bits */
    blk.qs[1] = 0x02;  /* code=2 at low 2 bits */
    unpack_tq2_0_block_to_ternary(blk, out);
    EXPECT_EQ(out[0], 0);  /* code 1 -> ternary 0 */
    EXPECT_EQ(out[1], 1);  /* code 2 -> ternary +1 */
    /* k=32: byte=(0)*32+(32&31)=0+0=0, shift=((32>>5)&3)*2=(1&3)*2=2.
     * qs[0] = 0x01, shift=2 -> 0 -> ternary -1. */
    EXPECT_EQ(out[32], -1);
}

/* --- compute_a_row_max_abs ----------------------------------------- */

void test_compute_a_row_max_abs() {
    std::vector<std::uint16_t> row(64);
    for (unsigned k = 0; k < 64; ++k) {
        const float v = (k < 32) ? float(k) : -(float)(k - 32);
        row[k] = f32_to_h(v);
    }
    /* max abs = max(31, 32) ; k=63 -> -31, k=32 -> 0 -> max abs is at
     * k=31 (positive 31) or k=63 (negative 31). All-zero excluded. */
    const float mx = compute_a_row_max_abs(row.data(), 64);
    /* abs values seen: 0, 1, ..., 31, 0, 1, ..., 31. Max = 31. */
    EXPECT_EQ(int(mx), 31);

    /* All-zero row: helper returns 1.0 to avoid divide-by-zero. */
    std::fill(row.begin(), row.end(), f32_to_h(0.0f));
    const float mx2 = compute_a_row_max_abs(row.data(), 64);
    EXPECT_EQ(int(mx2 * 1000), 1000);
}

/* --- INT4 packing tests (Phase 1 v4 primary path) ------------------ */

void test_ternary_to_int4_signed() {
    EXPECT_EQ(ternary_to_int4_signed(-1), 0xFu);
    EXPECT_EQ(ternary_to_int4_signed( 0), 0x0u);
    EXPECT_EQ(ternary_to_int4_signed(+1), 0x1u);
}

void test_pack_b_int4_invariants() {
    /* All-0 -> all dwords = 0. */
    std::vector<std::int8_t> ternary(64 * 16, 0);
    std::uint32_t out[128];
    pack_b_fragment_vnni_int4(ternary.data(), out);
    for (unsigned i = 0; i < 128; ++i) EXPECT_EQ(out[i], 0u);

    /* All +1 -> each dword = 0x11111111 (8 nibbles of 0001). */
    std::fill(ternary.begin(), ternary.end(), 1);
    pack_b_fragment_vnni_int4(ternary.data(), out);
    for (unsigned i = 0; i < 128; ++i) EXPECT_EQ(out[i], 0x11111111u);

    /* All -1 -> 0xFFFFFFFF. */
    std::fill(ternary.begin(), ternary.end(), -1);
    pack_b_fragment_vnni_int4(ternary.data(), out);
    for (unsigned i = 0; i < 128; ++i) EXPECT_EQ(out[i], 0xFFFFFFFFu);
}

void test_pack_b_int4_single_element() {
    /* ternary[k=5, n=3] = +1, rest = 0. */
    std::vector<std::int8_t> ternary(64 * 16, 0);
    ternary[5 * 16 + 3] = 1;
    std::uint32_t out[128];
    pack_b_fragment_vnni_int4(ternary.data(), out);
    /* k=5 -> d=0, bit_pos=(5%8)*4=20. Column n=3 -> out[3]. */
    EXPECT_EQ(out[3], (0x1u << 20));
    for (unsigned i = 0; i < 128; ++i) {
        if (i != 3) EXPECT_EQ(out[i], 0u);
    }

    /* k=63 -> d=7, bit_pos=(63%8)*4=28. n=15 -> out[7*16+15]=out[127]. */
    std::fill(ternary.begin(), ternary.end(), 0);
    ternary[63 * 16 + 15] = -1;
    pack_b_fragment_vnni_int4(ternary.data(), out);
    EXPECT_EQ(out[127], (0xFu << 28));
}

void test_pack_a_int4_quant() {
    /* a = 1.0 with s_a chosen so q = +7 (max). */
    std::vector<std::uint16_t> a(8 * 64, f32_to_h(1.0f));
    float s_a[8];
    for (unsigned m = 0; m < 8; ++m) s_a[m] = 1.0f;  /* a/s_a = 1, *7 = 7 */
    std::uint32_t out[64];
    pack_a_fragment_vnni_int4(a.data(), s_a, out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0x77777777u);

    /* a = 0.5 with s_a = 1.0 -> q = round(0.5 * 7) = 4. */
    std::fill(a.begin(), a.end(), f32_to_h(0.5f));
    pack_a_fragment_vnni_int4(a.data(), s_a, out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0x44444444u);

    /* Clamping: a = 5.0, s_a = 1.0 -> q = 35 -> clamped to 7. */
    std::fill(a.begin(), a.end(), f32_to_h(5.0f));
    pack_a_fragment_vnni_int4(a.data(), s_a, out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0x77777777u);

    /* Negative: a = -1.0 -> q = -7 -> bits 0x9 (= 1001 = -7 in 4-bit 2's comp). */
    std::fill(a.begin(), a.end(), f32_to_h(-1.0f));
    pack_a_fragment_vnni_int4(a.data(), s_a, out);
    for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(out[i], 0x99999999u);
}

int main() {
    test_ternary_to_int2_signed_roundtrip();
    test_pack_b_all_zero();
    test_pack_b_all_plus_one();
    test_pack_b_all_minus_one();
    test_pack_b_single_element();
    test_pack_b_dword_boundary();
    test_pack_b_last_dword();
    test_pack_a_all_zero();
    test_pack_a_all_plus_one_unit_scale();
    test_pack_a_quant_clamp();
    test_pack_a_per_row_scale();
    test_tq2_0_unpack_all_codes();
    test_compute_a_row_max_abs();
    /* INT4 path */
    test_ternary_to_int4_signed();
    test_pack_b_int4_invariants();
    test_pack_b_int4_single_element();
    test_pack_a_int4_quant();
    std::fprintf(stderr, "kv4_packing tests: %d pass, %d fail\n",
                 g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
