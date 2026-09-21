"""Release announcements require the exact tagged source and complete PyPI artifacts."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from urllib.error import HTTPError
import zipfile

import pytest

pytest.importorskip("tomllib", reason="Release tooling requires Python 3.11+")


_SPEC = importlib.util.spec_from_file_location(
    "release_metadata", Path(__file__).parents[1] / "scripts" / "release_metadata.py",
)
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)
VERSION = "0.4.6"
TAG = "v" + VERSION
COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network, git, or real retry sleep")

    monkeypatch.setattr(release, "urlopen", forbidden)
    monkeypatch.setattr(release, "git", forbidden)
    monkeypatch.setattr(release.time, "sleep", forbidden)


@pytest.fixture
def source():
    return {
        "pyproject.toml": b'[project]\nname = "programasweights"\nversion = "0.4.6"\n',
        "programasweights/__init__.py": b'__version__ = "0.4.6"\n',
        "programasweights/runtime.py": b"def infer(text):\n    return text\n",
        "PYPI_README.md": b"# PAW\n\nLocal inference.\n",
        "CHANGELOG.md": b"# Changes\n\n## 0.4.6 (2026-09-13)\n\n- Current change.\n\n## 0.4.5\n\n- Previous change.\n",
        ".github/workflows/test.yml": b"name: tests\n",
    }


def mock_tag(monkeypatch, source, *, tag_type=b"tag", remote_commit=COMMIT):
    ref = "refs/tags/" + TAG
    responses = {
        ("cat-file", "-t", ref): tag_type + b"\n",
        ("ls-remote", "--tags", "origin", ref, ref + "^{}"): (
            f"{'a' * 40}\t{ref}\n{remote_commit}\t{ref}^{{}}\n".encode()
        ),
        ("rev-parse", ref): ("a" * 40).encode() + b"\n",
        ("rev-parse", ref + "^{}"): COMMIT.encode() + b"\n",
        ("rev-parse", ref + "^{commit}"): COMMIT.encode() + b"\n",
        ("ls-tree", "-r", "--name-only", "-z", COMMIT): "\0".join(source).encode() + b"\0",
    }
    responses.update({("show", f"{COMMIT}:{path}"): data for path, data in source.items()})
    monkeypatch.setattr(release, "git", lambda *args: responses[args])


def artifact(source, kind, *, changed=None, omitted=(), version=VERSION,
             name="programasweights", description=None):
    description = source["PYPI_README.md"] if description is None else description
    metadata = (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        "Description-Content-Type: text/markdown\n\n"
    ).encode() + description
    members = dict(source) if kind == "sdist" else {
        path: data for path, data in source.items() if path.startswith("programasweights/")
    }
    members.update(changed or {})
    for path in omitted:
        members.pop(path)
    stream = io.BytesIO()
    if kind == "bdist_wheel":
        members[f"programasweights-{VERSION}.dist-info/METADATA"] = metadata
        with zipfile.ZipFile(stream, "w") as archive:
            for path, data in members.items():
                archive.writestr(path, data)
        filename = f"programasweights-{VERSION}-py3-none-any.whl"
    else:
        members["PKG-INFO"] = metadata
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path, data in members.items():
                member = tarfile.TarInfo(f"programasweights-{VERSION}/{path}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        filename = f"programasweights-{VERSION}.tar.gz"
    data = stream.getvalue()
    return data, {
        "filename": filename, "packagetype": kind, "yanked": False,
        "url": f"https://files.pythonhosted.org/packages/{filename}",
        "digests": {"sha256": hashlib.sha256(data).hexdigest()},
    }


def test_tagged_source_requires_the_published_annotated_tag(monkeypatch, source):
    mock_tag(monkeypatch, source)
    assert release.tagged_source(TAG) == (COMMIT, source)


@pytest.mark.parametrize("tag", ["0.4.6", "v0.4.6rc1", "main", "v0.4.6\n"])
def test_invalid_tag_fails_before_git(tag):
    with pytest.raises(ValueError, match="stable version tag"):
        release.tagged_source(tag)


def test_lightweight_tag_is_rejected(monkeypatch, source):
    mock_tag(monkeypatch, source, tag_type=b"commit")
    with pytest.raises(ValueError, match="annotated"):
        release.tagged_source(TAG)


def test_remote_tag_commit_mismatch_is_rejected(monkeypatch, source):
    mock_tag(monkeypatch, source, remote_commit="c" * 40)
    with pytest.raises(ValueError, match="published origin tag"):
        release.tagged_source(TAG)


def test_tag_must_match_package_version(monkeypatch, source):
    source["pyproject.toml"] = source["pyproject.toml"].replace(b"0.4.6", b"0.4.5")
    mock_tag(monkeypatch, source)
    with pytest.raises(ValueError, match="pyproject.toml versions differ"):
        release.tagged_source(TAG)


@pytest.mark.parametrize("initializer", [
    b'__version__ = "0.4.5"\n', b"# missing fallback\n",
    b'__version__ = "0.4.6"\n__version__ = "0.4.6"\n',
])
def test_fallback_version_must_match_unambiguously(monkeypatch, source, initializer):
    source["programasweights/__init__.py"] = initializer
    mock_tag(monkeypatch, source)
    with pytest.raises(ValueError, match="fallback version"):
        release.tagged_source(TAG)


def test_notes_include_only_the_requested_changelog_section(source):
    assert release.changelog_notes(source, VERSION) == (
        "- Current change.\n\n[PyPI 0.4.6](https://pypi.org/project/programasweights/0.4.6/)\n"
    )
    assert "Previous change" in release.changelog_notes(source, "0.4.5")


@pytest.mark.parametrize("changelog", [
    b"## 0.4.5\n\n- Wrong release.\n",
    b"## 0.4.6\n\n## 0.4.5\n\n- Old release.\n",
])
def test_missing_or_empty_release_notes_are_rejected(source, changelog):
    source["CHANGELOG.md"] = changelog
    with pytest.raises(ValueError, match="nonempty matching"):
        release.changelog_notes(source, VERSION)


def pypi_responses(monkeypatch, responses):
    pending = iter(responses)
    calls = []
    elapsed = [0]

    def open_response(url, timeout):
        calls.append(url)
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(release, "urlopen", open_response)
    monkeypatch.setattr(release.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(release.time, "sleep", lambda delay: elapsed.__setitem__(0, elapsed[0] + delay))
    return calls, elapsed


def pypi_json(files, version=VERSION):
    return {"info": {"version": version}, "urls": files}


def test_waits_through_404_and_partial_upload_until_both_artifacts_exist(monkeypatch, source):
    wheel = artifact(source, "bdist_wheel")[1]
    sdist = artifact(source, "sdist")[1]
    missing = HTTPError("https://pypi.org/test", 404, "Not Found", {}, None)
    calls, elapsed = pypi_responses(monkeypatch, [
        missing, pypi_json([wheel]), pypi_json([wheel, sdist]),
    ])
    assert release.pypi_release(VERSION, 30) == [wheel, sdist]
    assert len(calls) == 3
    assert elapsed[0] == 20


@pytest.mark.parametrize("missing_kind", ["bdist_wheel", "sdist"])
def test_incomplete_upload_is_rejected_at_deadline(monkeypatch, source, missing_kind):
    kind = "sdist" if missing_kind == "bdist_wheel" else "bdist_wheel"
    partial = pypi_json([artifact(source, kind)[1]])
    calls, elapsed = pypi_responses(monkeypatch, [partial, partial])
    with pytest.raises(ValueError, match="not both published"):
        release.pypi_release(VERSION, 3)
    assert len(calls) == 2
    assert elapsed[0] == 3


def test_unpublished_version_is_rejected_without_wait(monkeypatch):
    missing = HTTPError("https://pypi.org/test", 404, "Not Found", {}, None)
    calls, elapsed = pypi_responses(monkeypatch, [missing])
    with pytest.raises(ValueError, match="not both published"):
        release.pypi_release(VERSION, 0)
    assert len(calls) == 1
    assert elapsed[0] == 0


def test_non_404_http_error_is_not_hidden_by_retry(monkeypatch):
    error = HTTPError("https://pypi.org/test", 403, "Forbidden", {}, None)
    calls, elapsed = pypi_responses(monkeypatch, [error])
    with pytest.raises(HTTPError) as caught:
        release.pypi_release(VERSION, 30)
    assert caught.value is error
    assert len(calls) == 1
    assert elapsed[0] == 0


@pytest.mark.parametrize("problem", ["yanked", "wrong-version"])
def test_pypi_metadata_rejection_is_immediate(monkeypatch, source, problem):
    files = [artifact(source, kind)[1] for kind in ("bdist_wheel", "sdist")]
    if problem == "yanked":
        files[1]["yanked"] = True
    payload = pypi_json(files, "0.4.5" if problem == "wrong-version" else VERSION)
    calls, elapsed = pypi_responses(monkeypatch, [payload])
    with pytest.raises(ValueError, match="yanked|version differs"):
        release.pypi_release(VERSION, 30)
    assert len(calls) == 1
    assert elapsed[0] == 0


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
@pytest.mark.parametrize("readme", ["# PAW\n\nLocal inference.\n", "# PAW\n\nTiny functions — café, 日本語.\n"])
def test_exact_tagged_artifacts_accept_ascii_and_utf8_readmes(source, kind, readme):
    source["PYPI_README.md"] = readme.encode()
    data, metadata = artifact(source, kind)
    release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
def test_corrupt_download_hash_is_rejected_before_parsing(source, kind):
    data, metadata = artifact(source, kind)
    with pytest.raises(ValueError, match="SHA-256"):
        release.verify_artifact(data + b"corrupt", metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
def test_valid_hash_cannot_hide_source_mismatch(source, kind):
    data, metadata = artifact(source, kind, changed={"programasweights/runtime.py": b"# tampered\n"})
    with pytest.raises(ValueError, match="source differs"):
        release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
def test_missing_runtime_source_is_rejected(source, kind):
    data, metadata = artifact(source, kind, omitted=["programasweights/runtime.py"])
    with pytest.raises(ValueError, match="source differs|missing runtime"):
        release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
def test_untagged_runtime_source_is_rejected(source, kind):
    data, metadata = artifact(source, kind, changed={"programasweights/extra.py": b"# untagged\n"})
    with pytest.raises(ValueError, match="source differs|absent from the tag"):
        release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
@pytest.mark.parametrize("override", [{"version": "0.4.5"}, {"name": "another-package"}])
def test_wrong_artifact_identity_is_rejected(source, kind, override):
    data, metadata = artifact(source, kind, **override)
    with pytest.raises(ValueError, match="metadata does not match"):
        release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("kind", ["bdist_wheel", "sdist"])
def test_published_description_must_match_tag(source, kind):
    data, metadata = artifact(source, kind, description=b"Different description.\n")
    with pytest.raises(ValueError, match="description differs"):
        release.verify_artifact(data, metadata, source, VERSION)


@pytest.mark.parametrize("path", ["pyproject.toml", "PYPI_README.md", "CHANGELOG.md"])
def test_sdist_requires_packaging_inputs(source, path):
    data, metadata = artifact(source, "sdist", omitted=[path])
    with pytest.raises(ValueError, match="missing runtime or packaging"):
        release.verify_artifact(data, metadata, source, VERSION)


def test_sdist_may_omit_repository_only_files(source):
    data, metadata = artifact(source, "sdist", omitted=[".github/workflows/test.yml"])
    release.verify_artifact(data, metadata, source, VERSION)
