# Start Argonyx on the CraftifAI PipeGen pipeline (ARGONYX_DETECTOR=pipegen).
#
# PipeGen runs inside WSL2 (Ubuntu-22.04). Its MQTT detections reach Windows on
# localhost:1883, but its RTSP video (MediaMTX on port 8554) only plays when read
# from the WSL VM's own address -- measured: localhost / 127.0.0.1 open but never
# deliver a frame -- and that address changes whenever WSL restarts. So look it up
# each time, then point a camera named "PipeGen" at the stream.
#
#   powershell -ExecutionPolicy Bypass -File tools\run_pipegen.ps1
param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$StreamPath = "argonyx",
    # PipeGen's run.sh publishes to pipegen/detection/<source name>; the wildcard follows a source change.
    [string]$Topic = "pipegen/detection/#",
    [int]$Port = 8000,
    # Start only once PipeGen publishes. A file run lasts one pass of the clip, and
    # a camera started early backs off up to 30 s between retries and misses the start.
    [switch]$WaitForPipeGen
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$wslIp = ((wsl.exe -d $Distro -- hostname -I) -replace "`0", "").Trim().Split(" ")[0]
if (-not $wslIp) { throw "No IP for WSL distro $Distro - is it installed and running?" }
$rtsp = "rtsp://${wslIp}:8554/$StreamPath"

$env:ARGONYX_DETECTOR = "pipegen"
$env:ARGONYX_PIPEGEN_TOPIC = $Topic
Write-Host "Detections : mqtt://127.0.0.1:1883 (topic $Topic)"
Write-Host "Video      : $rtsp"

if ($WaitForPipeGen) {
    Write-Host "Waiting for PipeGen's first detection message - click Run in PipeGen now..."
    wsl.exe -d $Distro -- mosquitto_sub -h localhost -t $Topic -C 1 -W 900 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "No message from PipeGen on $Topic within 15 minutes" }
    Write-Host "PipeGen is publishing - starting Argonyx."
}

$server = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -WorkingDirectory $root -NoNewWindow -PassThru `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$Port"

$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
    try { Invoke-RestMethod "$base/api/cameras" | Out-Null; $ready = $true } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw "Argonyx did not come up on $base" }

$cam = @(Invoke-RestMethod "$base/api/cameras") | Where-Object { $_.name -eq "PipeGen" } | Select-Object -First 1
if ($cam) {
    Invoke-RestMethod -Method Post "$base/api/cameras/$($cam.id)/source" -ContentType "application/json" `
        -Body (@{ source = $rtsp } | ConvertTo-Json) | Out-Null
    Invoke-RestMethod -Method Post "$base/api/cameras/$($cam.id)/start" | Out-Null
} else {
    Invoke-RestMethod -Method Post "$base/api/cameras" -ContentType "application/json" `
        -Body (@{ name = "PipeGen"; source = $rtsp; start = $true } | ConvertTo-Json) | Out-Null
}
Write-Host "Control room: $base/dashboard  (Ctrl+C to stop)"
Wait-Process -Id $server.Id
