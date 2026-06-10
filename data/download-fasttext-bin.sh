#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

url="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
archive="cc.en.300.bin.gz"
model="cc.en.300.bin"

if [[ ! -f "$archive" ]]; then
  curl -L -C - --fail --progress-bar -o "$archive" "$url"
else
  echo "$archive already exists; skipping download."
fi

echo "Checking $archive..."
gzip -t "$archive"

if [[ -f "$model" ]]; then
  echo "$model already exists; skipping decompression."
else
  gzip -dk "$archive"
fi

echo "Ready: $(pwd)/$model"
