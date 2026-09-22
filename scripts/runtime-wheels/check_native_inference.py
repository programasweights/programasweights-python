"""Check offline CPU inference and native prefix-cache save/reload."""

import argparse
import json
import os
import sys
from pathlib import Path


def prohibit_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise AssertionError("Network use during offline inference: " + event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    os.environ["PAW_CACHE_DIR"] = str(args.cache_dir.resolve())
    os.environ["PAW_OFFLINE"] = "1"
    os.environ["GGML_METAL_DEVICES"] = "0"
    sys.addaudithook(prohibit_network)
    import llama_cpp
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
