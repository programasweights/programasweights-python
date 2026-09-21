# Releasing the Python SDK

A release has three separate records: an annotated Git tag identifies the source,
PyPI serves the installable wheel and source distribution, and a GitHub Release
shows the version and release notes on the repository. A tag or PyPI upload alone
does not create a GitHub Release. Finish and verify all three before reporting a
release complete.

## Publish a new version

1. Update `pyproject.toml`, the fallback `__version__` in
   `programasweights/__init__.py`, and the matching section in `CHANGELOG.md`.
   Commit and push these changes to `main` using the existing user Git identity.
   Release from a clean checkout of that pushed commit; confirm `HEAD` equals
   `origin/main` and the `tests` workflow is green for that exact commit.
2. Set the intended version, then build a wheel and source distribution into a
   new, empty directory. Install `build` and `twine` in the release environment if
   needed. Use the two exact filenames below, not a shared `dist/*` directory.

   ```bash
   SDK_RELEASE_VERSION=0.4.7  # replace with the version being released
   SDK_RELEASE_DIR=$(mktemp -d)
   python -m build --outdir "$SDK_RELEASE_DIR"
   python -m twine check \
     "$SDK_RELEASE_DIR/programasweights-$SDK_RELEASE_VERSION-py3-none-any.whl" \
     "$SDK_RELEASE_DIR/programasweights-$SDK_RELEASE_VERSION.tar.gz"
   ```

3. Create and push the annotated version tag **before uploading to PyPI**. Check
   that the remote tag resolves to the intended commit and is an annotated tag.
   Never replace an existing release tag with a new target.

   ```bash
   git tag -a "v$SDK_RELEASE_VERSION" -m "Release v$SDK_RELEASE_VERSION"
   git push origin "v$SDK_RELEASE_VERSION"
   git ls-remote --tags origin "refs/tags/v$SDK_RELEASE_VERSION*"
   ```

4. Upload the exact wheel and source distribution that passed `twine check`:

   ```bash
   python -m twine upload \
     "$SDK_RELEASE_DIR/programasweights-$SDK_RELEASE_VERSION-py3-none-any.whl" \
     "$SDK_RELEASE_DIR/programasweights-$SDK_RELEASE_VERSION.tar.gz"
   ```

5. The `release.yml` workflow starts on the `v*` tag push. It waits up to 600
   seconds for PyPI, verifies the remote annotated tag, version metadata, and
   published wheel/source distribution against the tagged source, then creates
   the GitHub Release with notes from that tag's `CHANGELOG.md` and the published
   artifacts. It preserves an existing release and marks a new release as latest
   only when its version is the latest on PyPI. If upload takes longer than the
   wait window, dispatch the workflow again after PyPI publication succeeds.
6. Verify the workflow succeeded, the GitHub Releases page has the expected entry
   and notes, and the latest-release entry agrees with PyPI's latest version.
   Compare both uploaded files' SHA-256 hashes with the version's PyPI JSON
   (`https://pypi.org/pypi/programasweights/<version>/json`, `urls[].digests.sha256`)
   and with the GitHub Release assets. Do not treat a tag listing as confirmation
   that the release entry exists.

## Recover a missing GitHub Release

When a version already exists on PyPI, keep its existing tag and published files.
Do not rebuild or republish that version, and never move its tag. Verify the
remote annotated tag, matching version metadata and changelog section, and the
published artifacts first. `scripts/release_metadata.py` requires Python 3.11+
and performs those checks, preparing the notes and assets in a new directory:

```bash
SDK_RELEASE_RECOVERY_DIR=$(mktemp -d)
git fetch origin --tags
python scripts/release_metadata.py --tag v0.4.6 \
  --output-dir "$SDK_RELEASE_RECOVERY_DIR/verified" --wait-seconds 600
gh workflow run release.yml --ref main -f tag=v0.4.6
```

Replace `v0.4.6` with the verified existing tag to repair. Run this from current
`main`, which contains the recovery workflow and script; the release content is
read from the requested tag. The workflow is idempotent: an existing GitHub
Release is preserved. Recheck the release entry, latest-version status, and
artifact hashes after it finishes. A source or artifact mismatch needs
investigation; do not retag or overwrite published artifacts to make it pass.
