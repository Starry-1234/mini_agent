$existing = [Environment]::GetEnvironmentVariable('Path', 'User')
$toAdd = 'F:\dev\AI_Tools\workspace\mini_agent\bin'
if ($existing -like "*$toAdd*") {
    Write-Output "PATH already contains: $toAdd"
    Write-Output "Current PATH: $existing"
} else {
    $new = "$existing;$toAdd"
    [Environment]::SetEnvironmentVariable('Path', $new, 'User')
    Write-Output "Added to user PATH: $toAdd"
    Write-Output "Note: open a NEW terminal for it to take effect (existing terminals keep old PATH)."
}
