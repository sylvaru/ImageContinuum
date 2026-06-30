param(
    [string]$Dxc = "dxc",
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$forwardSource = Join-Path $root "demo/res/shaders/forward/forward_bindless.hlsl"
$depthSource = Join-Path $root "demo/res/shaders/forward/depth_prepass.hlsl"
$computeTestSource = Join-Path $root "demo/res/shaders/forward/compute_test.hlsl"
$vulkanOut = Join-Path $root "demo/res/compiled_shaders/vulkan"
$dx12Out = Join-Path $root "demo/res/compiled_shaders/dx12"

New-Item -ItemType Directory -Force -Path $vulkanOut, $dx12Out | Out-Null

$commonFlags = @()
if ($Debug) {
    $commonFlags += @("-Zi", "-Qembed_debug")
} else {
    $commonFlags += @("-O3")
}
$commonFlags += @("-D", "IC_DISABLE_TEXTURE_SAMPLING=1")
$vulkanRegisterFlags = @(
    "-fvk-b-shift", "0", "0",
    "-fvk-t-shift", "1", "0",
    "-fvk-s-shift", "100", "0"
)

function Invoke-Dxc {
    param([string[]]$DxcArgs)

    & $Dxc @DxcArgs
    if ($LASTEXITCODE -ne 0) {
        throw "dxc failed with exit code $LASTEXITCODE"
    }
}

Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "vs_6_5", "-E", "VSMain", "-D", "IC_TARGET_DX12=1", "-Fo", (Join-Path $dx12Out "forward_bindless.vs.dxil"), $forwardSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "ps_6_5", "-E", "PSMain", "-D", "IC_TARGET_DX12=1", "-Fo", (Join-Path $dx12Out "forward_bindless.ps.dxil"), $forwardSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "vs_6_5", "-E", "VSMain", "-D", "IC_TARGET_DX12=1", "-Fo", (Join-Path $dx12Out "depth_prepass.vs.dxil"), $depthSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "ps_6_5", "-E", "PSMain", "-D", "IC_TARGET_DX12=1", "-Fo", (Join-Path $dx12Out "depth_prepass.ps.dxil"), $depthSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "cs_6_5", "-E", "CSMain", "-D", "IC_TARGET_DX12=1", "-D", "IC_VISIBILITY_TEST_WRITE=1", "-Fo", (Join-Path $dx12Out "visibility_test.cs.dxil"), $computeTestSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-T", "cs_6_5", "-E", "CSMain", "-D", "IC_TARGET_DX12=1", "-Fo", (Join-Path $dx12Out "independent_compute_test.cs.dxil"), $computeTestSource))

Invoke-Dxc -DxcArgs ($commonFlags + $vulkanRegisterFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "vs_6_6", "-E", "VSMain", "-D", "IC_TARGET_VULKAN=1", "-Fo", (Join-Path $vulkanOut "forward_bindless.vert.spv"), $forwardSource))
Invoke-Dxc -DxcArgs ($commonFlags + $vulkanRegisterFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "ps_6_6", "-E", "PSMain", "-D", "IC_TARGET_VULKAN=1", "-Fo", (Join-Path $vulkanOut "forward_bindless.frag.spv"), $forwardSource))
Invoke-Dxc -DxcArgs ($commonFlags + $vulkanRegisterFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "vs_6_6", "-E", "VSMain", "-D", "IC_TARGET_VULKAN=1", "-Fo", (Join-Path $vulkanOut "depth_prepass.vert.spv"), $depthSource))
Invoke-Dxc -DxcArgs ($commonFlags + $vulkanRegisterFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "ps_6_6", "-E", "PSMain", "-D", "IC_TARGET_VULKAN=1", "-Fo", (Join-Path $vulkanOut "depth_prepass.frag.spv"), $depthSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "cs_6_6", "-E", "CSMain", "-D", "IC_TARGET_VULKAN=1", "-D", "IC_VISIBILITY_TEST_WRITE=1", "-Fo", (Join-Path $vulkanOut "visibility_test.comp.spv"), $computeTestSource))
Invoke-Dxc -DxcArgs ($commonFlags + @("-spirv", "-fspv-target-env=vulkan1.3", "-T", "cs_6_6", "-E", "CSMain", "-D", "IC_TARGET_VULKAN=1", "-Fo", (Join-Path $vulkanOut "independent_compute_test.comp.spv"), $computeTestSource))
