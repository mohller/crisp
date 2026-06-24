#!/usr/bin/env bash
# Usage: bash recipe/update_sha256.sh 0.1.0
# Run this after pushing a release tag to GitHub.
set -euo pipefail

VERSION="${1:?usage: $0 <version>}"
URL="https://github.com/mohller/crisp/archive/refs/tags/v${VERSION}.tar.gz"

echo "Fetching ${URL} ..."
SHA256=$(curl -sL "${URL}" | sha256sum | awk '{print $1}')
echo "sha256: ${SHA256}"

sed -i "s|sha256: PLACEHOLDER|sha256: ${SHA256}|" "$(dirname "$0")/meta.yaml"
echo "Updated recipe/meta.yaml"
