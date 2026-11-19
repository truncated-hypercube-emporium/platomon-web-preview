# Build the browser version: web/dist/index.html (self-contained, wasm inlined).
# Needs: rustup target add wasm32-unknown-unknown
param([string]$TargetDir = $env:CARGO_TARGET_DIR)
# (Not "Stop": Windows PowerShell 5.1 turns cargo's progress on stderr into errors.)
$root = Split-Path -Parent $PSScriptRoot
if (-not $TargetDir) { $TargetDir = Join-Path $root "target" }

Push-Location $root
try {
  cargo build -p platomon-web --release --target wasm32-unknown-unknown --target-dir $TargetDir
  if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally { Pop-Location }

$wasm = Join-Path $TargetDir "wasm32-unknown-unknown/release/platomon_web.wasm"
$out = Join-Path $PSScriptRoot "dist"
New-Item -ItemType Directory -Force $out | Out-Null
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($wasm))
$tmpl = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "index.template.html"))
$html = $tmpl.Replace("__WASM_BASE64__", $b64)
[IO.File]::WriteAllText((Join-Path $out "index.html"), $html, (New-Object Text.UTF8Encoding($false)))
Copy-Item $wasm (Join-Path $out "platomon_web.wasm") -Force
"{0}  ({1:N0} KB, wasm {2:N0} KB)" -f (Join-Path $out "index.html"), ((Get-Item (Join-Path $out "index.html")).Length / 1KB), ((Get-Item $wasm).Length / 1KB)
