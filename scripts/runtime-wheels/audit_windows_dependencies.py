"""Check that a Windows runtime wheel bundles its non-system DLL dependencies."""

import argparse
import json
from pathlib import PurePosixPath
import zipfile

import pefile

# Windows 10+ components. MSVC and OpenMP redistributables are not OS libraries.
SYSTEM_DLLS = {"advapi32.dll", "kernel32.dll", "ntdll.dll", "ucrtbase.dll"}
IMPORT_TABLES = ("IMPORT", "DELAY_IMPORT")
REQUIRED_DLLS = {"llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    args = parser.parse_args()
    errors, report = [], {}
    with zipfile.ZipFile(args.wheel) as wheel:
        dlls = [name for name in wheel.namelist() if name.lower().endswith(".dll")]
        bundled = {
            PurePosixPath(name).name.lower()
            for name in dlls
            if str(PurePosixPath(name).parent) == "llama_cpp/lib"
        }
        for missing in sorted(REQUIRED_DLLS - bundled):
            errors.append("Missing llama_cpp/lib/" + missing)
        for name in sorted(dlls):
            with pefile.PE(data=wheel.read(name), fast_load=True) as binary:
                if binary.FILE_HEADER.Machine != pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
                    errors.append(name + ": not AMD64")
                binary.parse_data_directories(directories=[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_" + table]
                    for table in IMPORT_TABLES
                ])
                imports = {
                    table.lower(): sorted({entry.dll.decode("ascii").lower()
                        for entry in getattr(binary, "DIRECTORY_ENTRY_" + table, [])})
                    for table in IMPORT_TABLES
                }
                for warning in binary.get_warnings():
                    errors.append(name + ": " + warning)
                for dependencies in imports.values():
                    for dependency in dependencies:
                        if (dependency not in bundled and dependency not in SYSTEM_DLLS
                                and not dependency.startswith(("api-ms-win-", "ext-ms-win-"))):
                            errors.append(name + ": missing llama_cpp/lib/" + dependency)
                report[name] = imports
    print(json.dumps({"dlls": report, "errors": errors}, indent=2))
    if errors:
        raise SystemExit("Windows DLL dependency audit failed")


if __name__ == "__main__":
    main()
