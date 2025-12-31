# Create logs directory if it doesn't exist
if (-not (Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Generate a unique job ID (using timestamp)
$JOB_ID = Get-Date -Format "yyyyMMddHHmmss"
$LOG_FILE = "logs\${JOB_ID}_output.txt"

Write-Host "Starting job with ID: $JOB_ID"
Write-Host "Logs will be saved to: $LOG_FILE"

# Run the command in the background
# Start-Process is used to run the process detached.
# -WindowStyle Hidden runs it without a visible window (like a background job).
Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "scripts.averitec.evaluate", "--procedure_variant", "infact" `
    -RedirectStandardOutput $LOG_FILE `
    -RedirectStandardError $LOG_FILE `
    -WindowStyle Hidden

Write-Host "Job started in background."
