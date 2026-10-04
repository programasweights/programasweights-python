"""Check offline CPU inference and native prefix-cache save/reload."""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path


def prohibit_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise AssertionError("Network use during offline inference: " + event)


def check_linux_cpu_backend(llama_cpp, expected_variant=None):
    if sys.platform != "linux":
        return

    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]

    dynamic = ctypes.CDLL("libdl.so.2")
    dynamic.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dynamic.dladdr.restype = ctypes.c_int
    native = llama_cpp.llama_cpp._lib
    native.ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
    native.ggml_backend_reg_by_name.restype = ctypes.c_void_p
    native.ggml_backend_reg_get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    native.ggml_backend_reg_get_proc_address.restype = ctypes.c_void_p
    registry = native.ggml_backend_reg_by_name(b"CPU")
    assert registry, "CPU backend is not registered"
    function = native.ggml_backend_reg_get_proc_address(registry, b"ggml_backend_get_features")
    assert function, "Registered CPU backend lacks its feature function"
    location = DlInfo()
    assert dynamic.dladdr(function, ctypes.byref(location)) and location.filename, "Cannot locate CPU backend"
    module = Path(os.fsdecode(location.filename)).resolve()
    library_dir = (Path(llama_cpp.__file__).resolve().parent / "lib").resolve()
    assert module.parent == library_dir, (module, library_dir)
    assert module.name.startswith("libggml-cpu-") and module.name.endswith(".so"), module
    if expected_variant:
        assert module.name == "libggml-cpu-" + expected_variant + ".so", module
    print("CPU backend:", module.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--cpu-variant", choices=("x64",))
    args = parser.parse_args()
    os.environ["PAW_CACHE_DIR"] = str(args.cache_dir.resolve())
    os.environ["PAW_OFFLINE"] = "1"
    os.environ["GGML_METAL_DEVICES"] = "0"
    sys.addaudithook(prohibit_network)
    import llama_cpp
    if args.cpu_variant:
        assert sys.platform == "linux", "Explicit CPU variants require Linux"
        native = llama_cpp.llama_cpp._lib
        native.ggml_backend_reg_count.argtypes = []
        native.ggml_backend_reg_count.restype = ctypes.c_size_t
        assert native.ggml_backend_reg_count() == 0, "Backends were initialized before explicit loading"
        native.ggml_backend_load.argtypes = [ctypes.c_char_p]
        native.ggml_backend_load.restype = ctypes.c_void_p
        module = Path(llama_cpp.__file__).resolve().parent / "lib" / ("libggml-cpu-" + args.cpu_variant + ".so")
        assert native.ggml_backend_load(os.fsencode(module)), module
        assert native.ggml_backend_reg_count() == 1, "Expected one explicitly loaded backend"
    import programasweights as paw

    events = []
    for name in ("llama_state_seq_save_file", "llama_state_seq_load_file"):
        native = getattr(llama_cpp, name)

        def traced(*values, _native=native, _name=name):
            result = _native(*values)
            events.append((_name, int(result)))
            return result

        setattr(llama_cpp, name, traced)

    cases = [
        ("Urgent: production server is down. Please investigate now.", "immediate"),
        ("FYI: here is our monthly newsletter.", "wait"),
    ]
    manifest = Path(__file__).with_name("native-fixtures.json")
    for fixture in json.loads(manifest.read_text(encoding="utf-8")):
        program_id = fixture["program_id"]
        prefix = args.cache_dir / "programs" / program_id / "prefix_kv_cache.bin"
        prefix.unlink(missing_ok=True)
        for state in ("cold", "reloaded"):
            events.clear()
            with paw.function(program_id, offline=True, n_gpu_layers=0) as fn:
                assert fn._adapter is not None and fn._n_prefix > 0
                check_linux_cpu_backend(llama_cpp, args.cpu_variant)
                for text, expected in cases:
                    actual = fn(text)
                    assert actual == expected, (fixture["label"], state, text, actual)
            assert prefix.stat().st_size > 0
            operation = "save" if state == "cold" else "load"
            assert any(
                name == "llama_state_seq_" + operation + "_file" and size > 0
                for name, size in events
            ), events
            if state == "reloaded":
                assert not any(name == "llama_state_seq_save_file" for name, _ in events), events
            print("Passed", fixture["label"], state, flush=True)


if __name__ == "__main__":
    main()
