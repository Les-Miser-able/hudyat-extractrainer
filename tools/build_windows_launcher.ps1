# Build a small Windows GUI launcher with the compiler included in .NET Framework.
# No Python packages, .NET SDK, or administrator permissions are needed to build.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$source = Join-Path $PSScriptRoot 'windows\StudioLauncher.cs'
$output = Join-Path $projectRoot 'FSL Studio.exe'
$frameworkRoot = Join-Path $env:WINDIR 'Microsoft.NET'
$compiler = @(
    (Join-Path $frameworkRoot 'Framework64\v4.0.30319\csc.exe'),
    (Join-Path $frameworkRoot 'Framework\v4.0.30319\csc.exe')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $compiler) {
    throw 'The .NET Framework 4.x C# compiler was not found. Enable .NET Framework on this Windows installation.'
}

& $compiler /nologo /target:winexe /platform:anycpu /optimize+ `
    /reference:System.Windows.Forms.dll "/out:$output" $source
if ($LASTEXITCODE -ne 0) {
    throw "Launcher compilation failed (exit code $LASTEXITCODE). Close FSL Studio before rebuilding."
}

Write-Host "Built $output"
Write-Host 'Double-click FSL Studio.exe to open the trainer using .venv.'
