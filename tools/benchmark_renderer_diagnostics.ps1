param(
    [int]$Frames = 240,
    [int]$WarmupFrames = 200,
    [int]$WarmupSeconds = 3,
    [int]$Repetitions = 1,
    [string]$Output = "out/renderer-diagnostics-benchmark.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "out/build/windows-release/demo/ImageContinuumDemo.exe"
$outputPath = Join-Path $root $Output
$outputDirectory = Split-Path -Parent $outputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

$backends = @(
    @{ Name = "vulkan"; Config = "demo/res/configs/clustered_forward.toml" },
    @{ Name = "dx12"; Config = "demo/res/configs/clustered_forward_dx12.toml" }
)

# Section bits follow RendererDiagnostics::Section: Overview=1 through
# Backend=128. The baseline is repeated last to expose drift across the run.
$variants = @(
    @{ Name = "no_gui_prof_on"; Gui = 0; Diagnostics = -1; Profiler = 1; Async = 0 },
    @{ Name = "no_gui_prof_off"; Gui = 0; Diagnostics = -1; Profiler = 0; Async = 0 },
    @{ Name = "gui_closed"; Gui = 1; Diagnostics = 0; Profiler = 1; Async = 0 },
    @{ Name = "overview"; Gui = 1; Diagnostics = 1; Profiler = 1; Async = 0 },
    @{ Name = "async"; Gui = 1; Diagnostics = 2; Profiler = 1; Async = 0 },
    @{ Name = "queues"; Gui = 1; Diagnostics = 4; Profiler = 1; Async = 0 },
    @{ Name = "graph"; Gui = 1; Diagnostics = 8; Profiler = 1; Async = 0 },
    @{ Name = "passes"; Gui = 1; Diagnostics = 16; Profiler = 1; Async = 0 },
    @{ Name = "resources"; Gui = 1; Diagnostics = 32; Profiler = 1; Async = 0 },
    @{ Name = "visibility"; Gui = 1; Diagnostics = 64; Profiler = 1; Async = 0 },
    @{ Name = "backend"; Gui = 1; Diagnostics = 128; Profiler = 1; Async = 0 },
    @{ Name = "all"; Gui = 1; Diagnostics = 255; Profiler = 1; Async = 0 },
    @{ Name = "async_on_no_gui"; Gui = 0; Diagnostics = -1; Profiler = 1; Async = 1 },
    @{ Name = "no_gui_prof_on_repeat"; Gui = 0; Diagnostics = -1; Profiler = 1; Async = 0 }
)

Set-Content -LiteralPath $outputPath -Value (
    "# Renderer diagnostics benchmark {0:o}`n# Release, validation=off, vsync=off, frame limiter=off" -f (Get-Date))

for ($repetition = 1; $repetition -le $Repetitions; ++$repetition) {
    foreach ($variant in $variants) {
        # Alternating APIs inside each variant avoids measuring one API only at
        # the beginning/end of a run as clocks and background load drift.
        foreach ($backend in $backends) {
            $tag = "r$repetition-$($backend.Name)-$($variant.Name)"
            $env:IC_BENCH_FRAMES = [string]$Frames
            $env:IC_BENCH_WARMUP = [string]$WarmupFrames
            $env:IC_BENCH_WARMUP_S = [string]$WarmupSeconds
            $env:IC_BENCH_ASYNC = [string]$variant.Async
            $env:IC_BENCH_GUI = [string]$variant.Gui
            $env:IC_BENCH_VALIDATION = "0"
            $env:IC_BENCH_PROFILER = [string]$variant.Profiler
            $env:IC_BENCH_TAG = $tag
            if ($variant.Diagnostics -ge 0) {
                $env:IC_BENCH_DIAGNOSTICS = [string]$variant.Diagnostics
            } else {
                Remove-Item Env:IC_BENCH_DIAGNOSTICS -ErrorAction SilentlyContinue
            }

            $stdout = Join-Path $env:TEMP "$tag-out.txt"
            $stderr = Join-Path $env:TEMP "$tag-err.txt"
            Remove-Item -LiteralPath $stdout,$stderr -ErrorAction SilentlyContinue
            $process = Start-Process -FilePath $exe `
                -ArgumentList "--config",$backend.Config `
                -WorkingDirectory $root -WindowStyle Hidden `
                -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
                -PassThru -Wait
            if ($process.ExitCode -ne 0) {
                throw "$tag exited with code $($process.ExitCode)"
            }

            Get-Content -LiteralPath $stdout |
                Where-Object { $_ -match "\[Benchmark\] RESULT" } |
                Add-Content -LiteralPath $outputPath
            $errors = Get-Content -LiteralPath $stderr -ErrorAction SilentlyContinue
            if ($errors) {
                $errors | Add-Content -LiteralPath $outputPath
            }
        }
    }
}

Write-Output $outputPath
