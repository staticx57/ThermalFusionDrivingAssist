@echo off
REM ThermalFusionDrivingAssist - Windows Installation Script
REM For Windows 10/11 x86-64 systems (ThinkPad P16, etc.)
REM
REM This script installs:
REM - Python dependencies (OpenCV, NumPy, etc.)
REM - UVC webcam support (built-in to Windows)
REM - OPTIONAL: FLIR Firefly Spinnaker SDK (for global shutter cameras)
REM
REM Usage:
REM   install_windows.bat                 Install base system (UVC webcams)
REM   install_windows.bat --with-firefly  Install base + FLIR Firefly support
REM

setlocal enabledelayedexpansion

REM Colors for output (limited in CMD)
set "BLUE=[94m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

echo %BLUE%========================================%NC%
echo %BLUE%ThermalFusionDrivingAssist Installation%NC%
echo %BLUE%========================================%NC%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python not found!%NC%
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo %GREEN%Python version:%NC%
python --version
echo.

REM Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: pip not found!%NC%
    echo Please reinstall Python with pip enabled
    pause
    exit /b 1
)

REM Parse command-line arguments
set INSTALL_FIREFLY=false
if "%1"=="--with-firefly" set INSTALL_FIREFLY=true
if "%1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --with-firefly    Install FLIR Firefly Spinnaker SDK ^(optional^)
    echo   --help            Show this help message
    echo.
    exit /b 0
)

REM Step 1: Upgrade pip
echo %BLUE%[1/5] Upgrading pip...%NC%
python -m pip install --upgrade pip

REM Step 2: Install OpenCV
echo %BLUE%[2/5] Installing OpenCV...%NC%
python -m pip install opencv-python opencv-contrib-python

REM Step 3: Install NumPy and core dependencies
echo %BLUE%[3/5] Installing NumPy and core dependencies...%NC%
python -m pip install numpy

REM Step 4: Install project dependencies
echo %BLUE%[4/5] Installing project dependencies...%NC%
if exist requirements.txt (
    echo %GREEN%Found requirements.txt - installing...%NC%
    python -m pip install -r requirements.txt
) else (
    echo %YELLOW%requirements.txt not found. Installing core packages manually...%NC%
    python -m pip install numpy ultralytics supervision torch torchvision
)

REM Step 5: OPTIONAL - Install FLIR Firefly Spinnaker SDK
if "%INSTALL_FIREFLY%"=="true" (
    echo %BLUE%[5/5] Installing FLIR Firefly Spinnaker SDK...%NC%
    echo.
    echo %YELLOW%========================================%NC%
    echo %YELLOW%FLIR Spinnaker SDK Installation%NC%
    echo %YELLOW%========================================%NC%
    echo.
    echo %YELLOW%IMPORTANT: You need to manually download and install Spinnaker SDK:%NC%
    echo.
    echo %BLUE%Step 1: Download Spinnaker SDK%NC%
    echo   1. Go to: https://www.flir.com/products/spinnaker-sdk/
    echo   2. Create a free account / login
    echo   3. Download: SpinnakerSDK_FULL_3.2.0.62_x64.exe ^(or latest version^)
    echo   4. Run the installer
    echo.
    echo %BLUE%Step 2: Install Spinnaker SDK%NC%
    echo   1. Run SpinnakerSDK_FULL_3.2.0.62_x64.exe
    echo   2. Choose "Full Installation" ^(includes Python support^)
    echo   3. Accept license and complete installation
    echo   4. Restart your computer if prompted
    echo.
    echo %BLUE%Step 3: Install PySpin ^(Python wrapper^)%NC%
    echo   After installing Spinnaker SDK, PySpin should be available
    echo   Test with: python -c "import PySpin; print(PySpin.__version__)"
    echo.
    echo %YELLOW%Press any key when Spinnaker SDK installation is complete...%NC%
    pause >nul

    REM Verify PySpin installation
    python -c "import PySpin; print('PySpin version:', PySpin.__version__)" 2>nul
    if errorlevel 1 (
        echo %RED%ERROR: PySpin not detected!%NC%
        echo.
        echo %YELLOW%Troubleshooting:%NC%
        echo   1. Make sure you installed "Full Installation" of Spinnaker SDK
        echo   2. Check if PySpin wheel is in Spinnaker installation folder
        echo   3. Manually install: pip install spinnaker_python-X.X.X-py3-none-win_amd64.whl
        echo   4. Restart command prompt and try again
        echo.
        echo %YELLOW%You can still use UVC webcams without PySpin.%NC%
    ) else (
        echo %GREEN%SUCCESS: PySpin installed!%NC%
        echo %GREEN%FLIR Firefly cameras are now supported.%NC%
    )
) else (
    echo %BLUE%[5/5] Skipping FLIR Firefly support ^(use --with-firefly to enable^)%NC%
)

echo.
echo %GREEN%========================================%NC%
echo %GREEN%Installation Complete!%NC%
echo %GREEN%========================================%NC%
echo.
echo %GREEN%UVC webcam support: Ready%NC%
if "%INSTALL_FIREFLY%"=="true" (
    echo %GREEN%FLIR Firefly support: Check above for status%NC%
) else (
    echo %YELLOW%FLIR Firefly support: Not installed ^(use --with-firefly^)%NC%
)
echo.
echo %BLUE%Next steps:%NC%
echo   1. Connect your RGB camera ^(UVC webcam or FLIR Firefly^)
echo   2. Test camera: python camera_factory.py
echo   3. Run system: python main.py
echo.
echo %BLUE%Supported cameras:%NC%
echo   - Generic UVC webcams ^(Logitech, Microsoft, etc.^) - Works out-of-box
echo   - FLIR Firefly ^(global shutter^) - Requires Spinnaker SDK
echo.
echo %BLUE%For troubleshooting:%NC%
echo   - List cameras: python -c "from camera_factory import print_camera_summary; print_camera_summary()"
echo   - Test UVC: python rgb_camera_uvc.py
echo   - Test Firefly: python rgb_camera_firefly.py
echo.
echo %YELLOW%Press any key to exit...%NC%
pause >nul
