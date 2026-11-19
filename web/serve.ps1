# Minimal static server for web/dist (localhost only). The page also works
# opened straight from disk; this is for browsers or panes that need http.
param([int]$Port = 8765)
$dist = Join-Path $PSScriptRoot "dist"
$types = @{ ".html" = "text/html; charset=utf-8"; ".wasm" = "application/wasm"; ".js" = "text/javascript"; ".css" = "text/css" }
$l = New-Object Net.HttpListener
$l.Prefixes.Add("http://localhost:$Port/")
$l.Start()
"serving $dist on http://localhost:$Port/"
try {
  while ($l.IsListening) {
    $ctx = $l.GetContext()
    $path = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath.TrimStart("/"))
    if ($path -eq "") { $path = "index.html" }
    $file = [IO.Path]::GetFullPath((Join-Path $dist $path))
    $res = $ctx.Response
    if ($file.StartsWith($dist, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path $file -PathType Leaf)) {
      $bytes = [IO.File]::ReadAllBytes($file)
      $ext = [IO.Path]::GetExtension($file).ToLower()
      $res.ContentType = $(if ($types.ContainsKey($ext)) { $types[$ext] } else { "application/octet-stream" })
      $res.Headers.Add("Cache-Control", "no-store")
      $res.ContentLength64 = $bytes.Length
      $res.OutputStream.Write($bytes, 0, $bytes.Length)
    } else {
      $res.StatusCode = 404
    }
    $res.Close()
    "{0} {1} {2}" -f $ctx.Request.HttpMethod, $ctx.Request.Url.AbsolutePath, $res.StatusCode
  }
} finally { $l.Stop() }
