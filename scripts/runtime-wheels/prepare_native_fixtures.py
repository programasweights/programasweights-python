"""Prepare hash-pinned public programs and models without calling PAW services."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(asset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        with urlopen(asset["url"], timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        assert temporary.stat().st_size == asset["size_bytes"], temporary
        assert sha256(temporary) == asset["sha256"], temporary
        temporary.replace(path)
    assert path.stat().st_size == asset["size_bytes"], path
    assert sha256(path) == asset["sha256"], path
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    manifest = Path(__file__).with_name("native-fixtures.json")
    for fixture in json.loads(manifest.read_text(encoding="utf-8")):
        program_id = fixture["program_id"]
        bundle = download(fixture["bundle"], args.cache_dir / "bundles" / (program_id + ".paw"))
        download(fixture["base"], args.cache_dir / "base_models" / fixture["base"]["file"])
        program_dir = args.cache_dir / "programs" / program_id
        program_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle) as archive:
            assert set(archive.namelist()) == set(fixture["members"])
            for name, expected in fixture["members"].items():
                assert Path(name).name == name
                data = archive.read(name)
                assert hashlib.sha256(data).hexdigest() == expected, name
                (program_dir / name).write_bytes(data)
        metadata = json.loads((program_dir / "meta.json").read_text(encoding="utf-8"))
        assert metadata["program_id"] == program_id
        assert metadata["interpreter"] == fixture["interpreter"]
        print("Verified", fixture["label"], program_id, flush=True)


if __name__ == "__main__":
    main()
