# Define the virtual environment directory
$VENV_DIR = ".venv"

# Function to write colored output
function Write-Color {
    param (
        [string]$Message,
        [ConsoleColor]$Color
    )
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Message
    $Host.UI.RawUI.ForegroundColor = "White" # Reset to default color
}

# Check if virtual environment exists
if (-Not (Test-Path $VENV_DIR)) {
    Write-Color "Creating virtual environment..." "Yellow"
    python -m venv $VENV_DIR
}

# Activate the virtual environment
Write-Color "Activating virtual environment..." "Green"
# Activation for PowerShell
& "$VENV_DIR\bin\Activate.ps1"

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Color "Installing dependencies..." "Yellow"
    & "$VENV_DIR\bin\python.exe" -m pip install --upgrade pip
    & "$VENV_DIR\bin\python.exe" -m pip install -r requirements.txt
}

# Run the main script
Write-Color "Running main script..." "Cyan"
& "$VENV_DIR\bin\python.exe" main.py

# Deactivating the virtual environment
Write-Color "Deactivating virtual environment..." "Green"
# Deactivation is not required in script
