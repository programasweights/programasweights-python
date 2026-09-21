#!/usr/bin/env python3
"""Verify a tagged PyPI release and prepare its GitHub release assets (Python 3.11+)."""

import argparse
import ast
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
import tomllib
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile


def git(*args):
    return subprocess.check_output(["git", *args])


def tagged_source(tag):
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Use a stable version tag such as v0.4.6")
    ref = f"refs/tags/{tag}"
    if git("cat-file", "-t", ref).strip() != b"tag":
        raise ValueError("Release tags must be annotated")
    remote = dict(line.split()[::-1] for line in git("ls-remote", "--tags", "origin", ref, ref + "^{}").decode().splitlines())
    for name in (ref, ref + "^{}"):
        if remote.get(name) != git("rev-parse", name).decode().strip():
            raise ValueError("Local tag differs from the published origin tag")
    commit = git("rev-parse", ref + "^{commit}").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    source = {path: git("show", f"{commit}:{path}") for path in paths if path}
    version = tag[1:]
    if tomllib.loads(source["pyproject.toml"].decode())["project"]["version"] != version:
        raise ValueError("Tag and pyproject.toml versions differ")
    fallback = [node.value.value for node in ast.walk(ast.parse(source["programasweights/__init__.py"]))
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if fallback != [version]:
        raise ValueError("SDK fallback version does not match the tag")
    return commit, source


def changelog_notes(source, version):
    changelog = source["CHANGELOG.md"].decode()
    section = re.search(rf"^## {re.escape(version)}(?: \([^\n]+\))?\n(.*?)(?=^## |\Z)",
                        changelog, flags=re.MULTILINE | re.DOTALL)
    if not section or not section.group(1).strip():
        raise ValueError("Release needs a nonempty matching CHANGELOG.md section")
    return section.group(1).strip() + f"\n\n[PyPI {version}](https://pypi.org/project/programasweights/{version}/)\n"


def pypi_release(version, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with urlopen(f"https://pypi.org/pypi/programasweights/{version}/json", timeout=30) as response:
                metadata = json.load(response)
            files = metadata["urls"]
            if metadata["info"]["version"] != version:
                raise ValueError("PyPI version differs from the tag")
            if any(file.get("yanked") for file in files):
                raise ValueError("Refusing to announce a yanked release")
            if {file["packagetype"] for file in files} == {"bdist_wheel", "sdist"}:
                return files
        except HTTPError as exc:
            if exc.code != 404:
                raise
        if time.monotonic() >= deadline:
            raise ValueError("PyPI wheel and sdist are not both published; finish upload and rerun this workflow")
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def verify_artifact(data, file, source, version):
    if hashlib.sha256(data).hexdigest() != file["digests"]["sha256"]:
        raise ValueError("Downloaded artifact does not match its PyPI SHA-256")
    if file["packagetype"] == "bdist_wheel":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        actual = {name: content for name, content in files.items() if name.startswith("programasweights/")}
        expected = {name: content for name, content in source.items() if name.startswith("programasweights/")}
        metadata = files[f"programasweights-{version}.dist-info/METADATA"]
    elif file["packagetype"] == "sdist":
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            files = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                     for member in archive.getmembers() if member.isfile() and "/" in member.name}
        # Hatch may omit repository-only files. Every shipped source file must
        # match the tag; all runtime source and packaging inputs must be present.
        actual = {name: content for name, content in files.items() if name != "PKG-INFO"}
        if any(name not in source for name in actual):
            raise ValueError("Source archive contains files absent from the tag")
        required = {name for name in source if name.startswith("programasweights/")}
        required.update({"pyproject.toml", "PYPI_README.md", "CHANGELOG.md"})
        if not required.issubset(actual):
            raise ValueError("Source archive is missing runtime or packaging source")
        expected = {name: source[name] for name in actual}
        metadata = files["PKG-INFO"]
    else:
        raise ValueError("Unexpected distribution type")
    if actual != expected:
        raise ValueError("Published source differs from the release tag")
    parsed = BytesParser().parsebytes(metadata)
    if parsed["Name"] != "programasweights" or parsed["Version"] != version:
        raise ValueError("Artifact metadata does not match this release")
    # Core metadata uses UTF-8, without MIME Content-Type/charset headers.
    # BytesParser otherwise replaces non-ASCII characters in the body.
    description = re.split(r"\r?\n\r?\n", metadata.decode("utf-8"), maxsplit=1)
    if len(description) != 2 or description[1].strip() != source["PYPI_README.md"].decode().strip():
        raise ValueError("Published description differs from the tagged PyPI README")


def prepare(tag, output_dir, wait_seconds=0):
    commit, source = tagged_source(tag)
    version = tag[1:]
    notes = changelog_notes(source, version)
    files = pypi_release(version, wait_seconds)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    for file in files:
        filename = file["filename"]
        url = urlparse(file["url"])
        if Path(filename).name != filename or url.scheme != "https" or url.hostname != "files.pythonhosted.org":
            raise ValueError("Unexpected PyPI artifact filename or download host")
        with urlopen(file["url"], timeout=60) as response:
            data = response.read()
        verify_artifact(data, file, source, version)
        (artifacts / filename).write_bytes(data)
    with urlopen("https://pypi.org/pypi/programasweights/json", timeout=30) as response:
        latest = json.load(response)["info"]["version"] == version
    (output_dir / "notes.md").write_text(notes)
    result = {"tag": tag, "commit": commit, "latest": latest,
              "artifacts": {file["filename"]: file["digests"]["sha256"] for file in files}}
    (output_dir / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for verified artifacts/notes")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 600:
        parser.error("--wait-seconds must be between 0 and 600")
    print(json.dumps(prepare(args.tag, args.output_dir, args.wait_seconds)))


if __name__ == "__main__":
    main()
