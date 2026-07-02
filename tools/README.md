# Shader build tools

Double-click `build_shaders.bat` on Windows to compile every shader under
`demo/res/shaders` into:

- `demo/res/compiled_shaders/dx12`
- `demo/res/compiled_shaders/vulkan`

The builder discovers every `.hlsl` file that does not start with `_` and
compiles any `VSMain`, `PSMain`, or `CSMain` entry point it finds. Helper files
should use `.hlsli` or an underscore-prefixed `.hlsl` name.

Output names are based on the shader filename:

- `my_shader.hlsl` + `VSMain` -> `my_shader.vs.dxil` and `my_shader.vert.spv`
- `my_shader.hlsl` + `PSMain` -> `my_shader.ps.dxil` and `my_shader.frag.spv`
- `my_shader.hlsl` + `CSMain` -> `my_shader.cs.dxil` and `my_shader.comp.spv`

Optional command-line flags:

```bat
tools\build_shaders.bat --debug
tools\build_shaders.bat --dxc C:\Path\To\dxc.exe
```
