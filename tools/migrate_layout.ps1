# One-time migration from the original checkout layout. No data is overwritten.
[CmdletBinding()]
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$prefix = $projectRoot.TrimEnd('\') + '\'
function Checked-Path([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the project: $absolute"
    }
    return $absolute
}
$moves = @(
    @('datasets', 'data/processed/default'),
    @('datasets_augmented', 'data/processed/augmented'),
    @('models/hand_landmarker.task', 'assets/hand_landmarker.task'),
    @('augmentation/configs', 'configs/augmentation'),
    @('augmentation/transforms', 'src/fsl_trainer/preparation/augmentation'),
    @('augmentation/tests', 'tests/augmentation'),
    @('augmentation/previews', 'reports/augmentation/previews'),
    @('augmentation/reports', 'reports/augmentation/reports'),
    @('augmentation/README.md', 'configs/augmentation/README.md')
)
# Enumerate only legacy category folders, excluding the new data layout.
Get-ChildItem -LiteralPath (Checked-Path 'data') -Directory -Force | ForEach-Object {
    if ($_.Name -notin @('raw', 'processed', 'cache')) {
        $moves += ,@(('data/' + $_.Name), ('data/raw/' + $_.Name))
    }
}
$pending = @()
foreach ($move in $moves) {
    $source = Checked-Path $move[0]
    $target = Checked-Path $move[1]
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $target) { throw "Destination collision: $target" }
        $pending += [PSCustomObject]@{Source=$source; Target=$target}
    }
}
function Inventory([string]$path) {
    $item = Get-Item -LiteralPath $path -Force
    $entries = @($item)
    if ($item.PSIsContainer) {
        $entries += @(Get-ChildItem -LiteralPath $path -Recurse -Force)
    }
    foreach ($entry in $entries) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Linked path is unsupported during migration: $($entry.FullName)"
        }
        $relative = $entry.FullName.Substring($path.Length)
        if ($entry.PSIsContainer) { "D|$relative" }
        else { "F|$relative|$($entry.Length)|$($entry.LastWriteTimeUtc.Ticks)" }
    }
}
# Inventory every source before any mutation, rejecting links and collisions first.
$inventories = @{}
foreach ($move in $pending) {
    $inventories[$move.Source] = @(Inventory $move.Source | Sort-Object)
    Write-Output "$($move.Source) -> $($move.Target) ($($inventories[$move.Source].Count) entries)"
}
if ($DryRun) { return }
foreach ($move in $pending) {
    New-Item -ItemType Directory -Path (Split-Path $move.Target) -Force | Out-Null
    Move-Item -LiteralPath $move.Source -Destination $move.Target
    $after = @(Inventory $move.Target | Sort-Object)
    if (Compare-Object $inventories[$move.Source] $after) {
        throw "Inventory mismatch after moving to $($move.Target); stop and inspect."
    }
}
# Remove only empty obsolete scaffold directories, never recursively.
foreach ($relative in @('models', 'augmentation')) {
    $directory = Checked-Path $relative
    if ((Test-Path -LiteralPath $directory) -and
        -not (Get-ChildItem -LiteralPath $directory -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $directory
    }
}
Write-Output "Verified $($pending.Count) moves; raw bytes and existing metadata were not rewritten."
