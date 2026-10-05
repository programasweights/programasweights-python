"""Install a runtime wheel and verify offline inference with this Python."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--check-x64", action="store_true", help="Also verify the Linux x64 backend.")
    args = parser.parse_args()
    if args.check_x64 and sys.platform != "linux":
        parser.error("--check-x64 requires Linux")
    wheel = args.wheel.resolve()
    scripts = Path(__file__).resolve().parent
    checkout = scripts.parents[1]
    with tempfile.TemporaryDirectory() as root:
        venv.create(root, with_pip=True)
        python = str(Path(root) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        fixtures = str(Path(root) / "fixtures")
        commands = [
            ["-m", "pip", "install", str(wheel), str(checkout)],
            ["-m", "pip", "check"],
            ["-I", "-c", "import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"],
            ["-I", str(scripts / "prepare_native_fixtures.py"), fixtures],
            ["-I", str(scripts / "check_native_inference.py"), fixtures],
        ]
        if args.check_x64:
            commands.append([
                "-I", str(scripts / "check_native_inference.py"), fixtures, "--cpu-variant", "x64",
            ])
        for command in commands:
            subprocess.run([python, *command], check=True, cwd=checkout)


if __name__ == "__main__":
    main()
