"""Check scorer OS-state gates using injected hardware/kernel responses."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).read_text()
start = "static int ggml_backend_cpu_x86_score() {"
end = "GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)"
assert source.count(start) == source.count(end) == 1
scorer = start + source.split(start, 1)[1].split(end, 1)[0]
xgetbv = '    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));'
assert scorer.count(xgetbv) == 1
assert scorer.count("syscall(SYS_arch_prctl, ") == 3
scorer = scorer.replace(xgetbv, "    test_xgetbv(eax, edx);")
scorer = scorer.replace("syscall(SYS_arch_prctl, ", "test_arch_prctl(")
variants = {"x64": [], "avx": ["AVX"], "avx512": ["AVX", "AVX512"],
            "amx": ["AVX", "AVX512", "AMX_TILE", "AMX_INT8"]}
tile = (1 << 17) | (1 << 18)
full = tile | 0xe7
defaults = [0, full, tile, tile, 0]
cases = [("all available", defaults, set(variants))]

def add_case(name, index, value, eligible):
    inputs = defaults.copy()
    inputs[index] = value
    cases.append((name, inputs, set(eligible)))

for missing, name in enumerate(("AVX", "XSAVE", "OSXSAVE"), 1):
    add_case("missing " + name, 0, missing, ["x64"])
for bit in (1, 2, 5, 6, 7, 17, 18):
    eligible = ["x64"] if bit < 3 else ["x64", "avx"]
    if bit > 7:
        eligible.append("avx512")
    add_case("missing XCR0 bit " + str(bit), 1, full & ~(1 << bit), eligible)
without_amx = ["x64", "avx", "avx512"]
add_case("missing AMX_TILE", 0, 4, without_amx)
for index, name in ((2, "kernel support"), (3, "kernel permission")):
    for bit in (17, 18):
        add_case(name + " missing bit " + str(bit), index, tile & ~(1 << bit), without_amx)
for request in (0x1021, 0x1022, 0x1023):
    add_case("failed arch_prctl " + hex(request), 4, request, without_amx)

with tempfile.TemporaryDirectory(prefix="cpu-eligibility-") as temporary:
    output = Path(temporary)
    (output / "scorer-under-test.inc").write_text(scorer)
    for name, features in variants.items():
        library = output / (name + ".so")
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O2", "-shared", "-fPIC", "-fno-lto",
            "-march=x86-64", "-mtune=generic", "-I", str(output),
            *["-DGGML_" + feature for feature in features],
            str(Path(__file__).with_name("linux_eligibility_probe.cpp")), "-o", str(library)]
        subprocess.run(command, check=True)
        score = ctypes.CDLL(str(library)).score_case
        score.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
        score.restype = ctypes.c_int
        for case, inputs, eligible in cases:
            actual = score(*inputs)
            assert (actual > 0) == (name in eligible), (name, case, actual)
        print("Passed", name, len(cases), "OS-state eligibility cases", flush=True)
