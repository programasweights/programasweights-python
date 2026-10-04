#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
download_dir=$(mktemp -d)
trap 'rm -rf -- "$download_dir"' EXIT
base_url=https://vault.centos.org/7.9.2009/sclo/x86_64/rh/Packages/d

while read -r expected_sha256 package; do
    curl --fail --location --retry 3 \
        --output "$download_dir/$package" "$base_url/$package"
done < "$script_dir/linux-toolchain.sha256"

cd "$download_dir"
sha256sum --check "$script_dir/linux-toolchain.sha256"
rpm -Uvh ./*.rpm
