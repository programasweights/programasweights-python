#!/usr/bin/env bash
set -euo pipefail

version=${2:-0.3.36}
case "$version" in
    0.3.20|0.3.21|0.3.22|0.3.23|0.3.24|0.3.25|0.3.26|0.3.27|0.3.28|0.3.29|0.3.30|0.3.31|0.3.32|0.3.33|0.3.34|0.3.36)
        patches=(linux-backend-install-dir linux-backend-packaging
                 linux-backend-search-path linux-cpu-os-state
                 linux-amx-permission linux-amx-gcc11)
        if [[ "$version" == 0.3.20 ]]; then
            patches+=(linux-backend-init)
        elif [[ "$version" == 0.3.21 || "$version" == 0.3.22 ]]; then
            patches+=(linux-backend-init)
        elif [[ "$version" == 0.3.27 ]]; then
            patches+=(llama-cpp-ext-nextn)
        elif [[ "$version" == 0.3.36 ]]; then
            patches+=(linux-q6k-avx512)
        fi
        ;;
    *)
        printf 'Unsupported Linux runtime version: %s\n' "$version" >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "${1:-wheelhouse}"
wheelhouse=$(cd -- "${1:-wheelhouse}" && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT

read -r source_spec < "$script_dir/sources/$version.txt"
curl --fail --location --retry 3 --output "$build_dir/source.tar.gz" "${source_spec%%#sha256=*}"
printf '%s  %s\n' "${source_spec##*#sha256=}" "$build_dir/source.tar.gz" | sha256sum --check
tar -xzf "$build_dir/source.tar.gz" -C "$build_dir"
cd "$build_dir/llama_cpp_python-$version"
for patch_name in "${patches[@]}"; do
    patch --batch --fuzz=0 -p1 < "$script_dir/$patch_name.patch"
done

bash "$script_dir/install-linux-toolchain.sh"
export PATH=/opt/python/cp38-cp38/bin:/opt/rh/devtoolset-11/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rh/devtoolset-11/root/usr/lib64:/opt/rh/devtoolset-11/root/usr/lib:/usr/local/lib64
export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++
export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export SSL_CERT_FILE=/opt/_internal/certs.pem
export CMAKE_GENERATOR='Unix Makefiles'
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
export CMAKE_ARGS="-C $script_dir/linux-dispatch.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
export SKBUILD_BUILD_DIR="$build_dir/build"

python "$script_dir/check_linux_cpu_eligibility.py" \
    vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp

python -m pip install -r "$script_dir/linux-requirements.txt"
python -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
    . --wheel-dir "$build_dir/raw-wheels"
python - "$build_dir/build/compile_commands.json" <<'PY'
import json, pathlib, re, shlex, sys
commands = json.loads(pathlib.Path(sys.argv[1]).read_text())
variants = (
    "x64 sse42 sandybridge ivybridge piledriver haswell skylakex "
    "cannonlake cascadelake icelake cooperlake zen4 alderlake sapphirerapids"
).split()
optimized = {"ggml-cpu-" + name for name in variants if name != "x64"}
baseline = {"-march=x86-64", "-mtune=generic"}
scorers = []
for command in commands:
    args = command.get("arguments") or shlex.split(command["command"])
    assert not any(arg.startswith("@") for arg in args), command
    assert not any(arg in ("-march=native", "-mtune=native", "-mcpu=native") or arg.startswith("-flto")
                   for arg in args), command
    output = args[args.index("-o") + 1]
    target = re.search(r"(?:^|/)CMakeFiles/([^/]+)\.dir/", output).group(1)
    if target not in optimized:
        assert baseline <= set(args), command
        assert all(arg in baseline for arg in args if arg.startswith("-m")), command
    if command["file"].endswith("/arch/x86/cpu-feats.cpp"):
        scorers.append(target)
        assert "-fno-lto" in args, command
assert sorted(scorers) == sorted("ggml-cpu-" + name + "-feats" for name in variants), scorers
PY
auditwheel repair --plat manylinux2014_x86_64 --only-plat \
    --wheel-dir "$wheelhouse" "$build_dir"/raw-wheels/*.whl
