// Test only: the generated include contains the actual patched scorer function.
#include <cstdint>
#include <cstdlib>

enum MissingFeature { NONE, CPU_AVX, CPU_XSAVE, CPU_OSXSAVE, CPU_AMX_TILE };
static int missing_feature, failed_request;
static uint64_t xcr0_value, supported_state, permitted_state;

struct cpuid_x86 {
    bool AVX() { return missing_feature != CPU_AVX; }
    bool XSAVE() { return missing_feature != CPU_XSAVE; }
    bool OSXSAVE() { return missing_feature != CPU_OSXSAVE; }
    bool AMX_TILE() { return missing_feature != CPU_AMX_TILE; }
    bool AMX_INT8() { return true; }
    bool AVX512F() { return true; }
    bool AVX512CD() { return true; }
    bool AVX512VL() { return true; }
    bool AVX512DQ() { return true; }
    bool AVX512BW() { return true; }
};

static void test_xgetbv(uint32_t & eax, uint32_t & edx) {
    // A regressed guard must fail even on a host where XGETBV would succeed.
    if (missing_feature >= CPU_AVX && missing_feature <= CPU_OSXSAVE) {
        std::abort();
    }
    eax = uint32_t(xcr0_value);
    edx = uint32_t(xcr0_value >> 32);
}

static long test_arch_prctl(int request, uint64_t * state) {
    if (request != 0x1021 && request != 0x1022) { std::abort(); }
    *state = request == 0x1021 ? supported_state : permitted_state;
    return request == failed_request ? -1 : 0;
}

static long test_arch_prctl(int request, int feature) {
    if (request != 0x1023 || feature != 18) { std::abort(); }
    return request == failed_request ? -1 : 0;
}

#include "scorer-under-test.inc"

extern "C" int score_case(int missing, uint64_t xcr0, uint64_t supported,
                          uint64_t permitted, int denied_request) {
    missing_feature = missing;
    xcr0_value = xcr0;
    supported_state = supported;
    permitted_state = permitted;
    failed_request = denied_request;
    return ggml_backend_cpu_x86_score();
}
