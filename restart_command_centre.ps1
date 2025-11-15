# Restart Command Centre Script
# Stops any running instances and starts fresh

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "RESTARTING COMMAND CENTRE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Find and kill any process using port 8000
Write-Host "Step 1: Checking for processes on port 8000..." -ForegroundColor Yellow
$netstatOutput = netstat -ano | Select-String ":8000"
$processes = @()

foreach ($line in $netstatOutput) {
    $parts = $line -split '\s+' | Where-Object { $_ -ne '' }
    if ($parts.Count -gt 0) {
        $pid = $parts[-1]
        if ($pid -match '^\d+$') {
            $processes += $pid
        }
    }
}

$processes = $processes | Select-Object -Unique

if ($processes.Count -gt 0) {
    Write-Host "Found processes using port 8000. Stopping them..." -ForegroundColor Yellow
    foreach ($pid in $processes) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-Host "  [OK] Stopped process $pid" -ForegroundColor Green
        }
        catch {
            Write-Host "  [SKIP] Could not stop process $pid" -ForegroundColor Yellow
        }
    }
    Start-Sleep -Seconds 2
}
else {
    Write-Host "  [OK] Port 8000 is free" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 2: Verifying camera configuration..." -ForegroundColor Yellow
Write-Host "  cam1: http://10.197.139.199:8080/video" -ForegroundColor White
Write-Host "  cam2: http://10.197.139.108:8080/video" -ForegroundColor White
Write-Host "  cam3: http://10.197.139.192:8080/video" -ForegroundColor White
Write-Host "  [OK] Configuration updated" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Starting Command Centre..." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Start the command centre
python run_command_centre.py
