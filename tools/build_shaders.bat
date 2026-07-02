@echo off
setlocal
pushd "%~dp0\.."

node tools\build_shaders.js %*
if errorlevel 1 (
    echo.
    echo Shader build failed.
    pause
    popd
    exit /b 1
)

echo.
echo Shader build succeeded.
pause
popd
