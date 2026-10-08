#!/bin/sh
set -eu
native_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:-"$native_dir/../build/ane-bridge"}
mkdir -p "$(dirname -- "$output")"
xcrun swiftc -O -target arm64-apple-macos15.0 "$native_dir/bridge.swift" -o "$output" -framework CoreML -framework Foundation
printf 'Built %s\n' "$output"
