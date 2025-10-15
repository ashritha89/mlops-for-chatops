# Download Python 3.11
$pythonUrl = "https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
$outputPath = "$env:TEMP\python-3.11.5-amd64.exe"

Write-Host "Downloading Python 3.11..."
Invoke-WebRequest -Uri $pythonUrl -OutFile $outputPath

# Install Python with pip
Write-Host "Installing Python 3.11..."
Start-Process -FilePath $outputPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1" -Wait

# Verify installation
Write-Host "Verifying Python installation..."
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
python --version
pip --version