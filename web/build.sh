#!/usr/bin/env bash
# Build the browser version: web/dist/index.html (self-contained, wasm inlined).
# Bash counterpart of build.ps1, for Linux/macOS and the GitHub Pages workflow.
# Needs: rustup target add wasm32-unknown-unknown, python3
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
target_dir="${CARGO_TARGET_DIR:-$root/target}"

cargo build --manifest-path "$root/Cargo.toml" -p platomon-web --release \
  --target wasm32-unknown-unknown --target-dir "$target_dir"

wasm="$target_dir/wasm32-unknown-unknown/release/platomon_web.wasm"
out="$here/dist"
mkdir -p "$out"

python3 - "$wasm" "$here/index.template.html" "$out/index.html" <<'PY'
import base64, sys
wasm, tmpl, dst = sys.argv[1:4]
b64 = base64.b64encode(open(wasm, "rb").read()).decode("ascii")
html = open(tmpl, encoding="utf-8").read().replace("__WASM_BASE64__", b64)
open(dst, "w", encoding="utf-8", newline="").write(html)
PY

cp "$wasm" "$out/platomon_web.wasm"
# Serve files as-is on GitHub Pages (no Jekyll processing).
touch "$out/.nojekyll"

printf '%s  (%d KB, wasm %d KB)\n' "$out/index.html" \
  $(( $(wc -c < "$out/index.html") / 1024 )) $(( $(wc -c < "$wasm") / 1024 ))
