# Innovative-Approaches-to-Asset-Prediction

## Running the Project

### Prerequisites

Make sure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/).

### Setting Up the Environment

You can set up and run the project using either a PowerShell script (`run.ps1`) on Windows or a Bash script (`run.sh`) on Unix-like systems.

#### Using PowerShell (`run.ps1`) on Windows

1. **Open PowerShell** and navigate to the project directory:
    ```powershell
    cd path\to\Innovative-Approaches-to-Asset-Prediction
    ```

2. **Set the execution policy** to allow running scripts if it's not already set:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

3. **Run the PowerShell script**:
    ```powershell
    .\run.ps1
    ```

#### Using Bash (`run.sh`) on Unix-like Systems

1. **Open a terminal** and navigate to the project directory:
    ```bash
    cd path/to/Innovative-Approaches-to-Asset-Prediction
    ```

2. **Make the script executable** (if it is not already):
    ```bash
    chmod +x run.sh
    ```

3. **Run the Bash script**:
    ```bash
    ./run.sh
    ```

### What the Scripts Do

- **Create a virtual environment** in the project directory (if it doesn't already exist).
- **Activate the virtual environment**.
- **Install the required dependencies** listed in `requirements.txt`.
- **Run the main script** (`main.py`).

These scripts automate the setup and execution process, making it easier to get the project up and running.

### Additional Notes

- Ensure that your terminal or PowerShell has the necessary permissions to execute scripts.
- If you encounter any issues with the execution policy in PowerShell, you might need to run PowerShell as an administrator to change the execution policy.
- The virtual environment directory is named `.venv` and is located in the project root directory.
