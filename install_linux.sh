#!/bin/bash
#
# ThermalFusionDrivingAssist - Linux Installation Script
# For Jetson Orin Nano (ARM64) and x86-64 Linux systems
#
# This script installs:
# - System dependencies (OpenCV, Python packages)
# - UVC webcam support (built-in to Linux kernel)
# - OPTIONAL: FLIR Firefly Spinnaker SDK (for global shutter cameras)
#
# Usage:
#   chmod +x install_linux.sh
#   ./install_linux.sh                 # Install base system (UVC webcams)
#   ./install_linux.sh --with-firefly  # Install base + FLIR Firefly support
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect architecture
ARCH=$(uname -m)
IS_JETSON=false

if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ThermalFusionDrivingAssist Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Architecture: ${GREEN}${ARCH}${NC}"
if [ "$IS_JETSON" = true ]; then
    echo -e "Platform: ${GREEN}NVIDIA Jetson$(cat /etc/nv_tegra_release)${NC}"
else
    echo -e "Platform: ${GREEN}Generic Linux${NC}"
fi
echo ""

# Parse command-line arguments
INSTALL_FIREFLY=false
for arg in "$@"; do
    case $arg in
        --with-firefly)
            INSTALL_FIREFLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-firefly    Install FLIR Firefly Spinnaker SDK (optional)"
            echo "  --help            Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# Step 1: Update system packages
echo -e "${BLUE}[1/6] Updating system packages...${NC}"
sudo apt-get update

# Step 2: Install Python dependencies
echo -e "${BLUE}[2/6] Installing Python dependencies...${NC}"
sudo apt-get install -y python3 python3-pip python3-dev python3-numpy

# Step 3: Install OpenCV (if not already installed)
echo -e "${BLUE}[3/6] Installing OpenCV...${NC}"
if [ "$IS_JETSON" = true ]; then
    # Jetson comes with OpenCV pre-installed (with CUDA support)
    echo -e "${GREEN}✓ Jetson platform detected - using pre-installed OpenCV with CUDA${NC}"
    python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || {
        echo -e "${YELLOW}⚠ OpenCV not found. Installing...${NC}"
        sudo apt-get install -y python3-opencv
    }
else
    # Generic Linux - install OpenCV via pip
    echo -e "${YELLOW}Installing OpenCV via pip...${NC}"
    pip3 install --user opencv-python opencv-contrib-python
fi

# Step 4: Install V4L2 utilities (for UVC webcam debugging)
echo -e "${BLUE}[4/6] Installing V4L2 utilities...${NC}"
sudo apt-get install -y v4l-utils

# List available cameras
echo -e "${GREEN}✓ Listing available UVC cameras:${NC}"
v4l2-ctl --list-devices || echo -e "${YELLOW}⚠ No cameras detected${NC}"

# Step 5: Install project dependencies
echo -e "${BLUE}[5/6] Installing project Python dependencies...${NC}"
if [ -f requirements.txt ]; then
    pip3 install --user -r requirements.txt
else
    echo -e "${YELLOW}⚠ requirements.txt not found. Installing core packages manually...${NC}"
    pip3 install --user numpy ultralytics supervision torch torchvision
fi

# Step 6: OPTIONAL - Install FLIR Firefly Spinnaker SDK
if [ "$INSTALL_FIREFLY" = true ]; then
    echo -e "${BLUE}[6/6] Installing FLIR Firefly Spinnaker SDK...${NC}"

    # Check architecture
    if [ "$ARCH" = "aarch64" ]; then
        # ARM64 (Jetson)
        SPINNAKER_ARCH="arm64"
        SPINNAKER_VERSION="spinnaker-3.2.0.62-Ubuntu22.04-arm64-pkg.tar.gz"  # Latest as of 2025
        SPINNAKER_URL="https://flir.netx.net/file/asset/54321/original/attachment"  # Placeholder - requires FLIR account
    elif [ "$ARCH" = "x86_64" ]; then
        # x86-64
        SPINNAKER_ARCH="amd64"
        SPINNAKER_VERSION="spinnaker-3.2.0.62-Ubuntu22.04-amd64-pkg.tar.gz"  # Latest as of 2025
        SPINNAKER_URL="https://flir.netx.net/file/asset/54320/original/attachment"  # Placeholder - requires FLIR account
    else
        echo -e "${RED}✗ Unsupported architecture for Spinnaker: ${ARCH}${NC}"
        exit 1
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}FLIR Spinnaker SDK Installation${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: You need to manually download Spinnaker SDK from FLIR website:${NC}"
    echo -e "${BLUE}  1. Go to: https://www.flir.com/products/spinnaker-sdk/${NC}"
    echo -e "${BLUE}  2. Create a free account / login${NC}"
    echo -e "${BLUE}  3. Download: ${SPINNAKER_VERSION}${NC}"
    echo -e "${BLUE}  4. Place the .tar.gz file in the current directory${NC}"
    echo ""
    echo -e "${YELLOW}Once downloaded, we'll install it automatically.${NC}"
    echo ""

    # Check if Spinnaker tarball exists
    if [ -f "${SPINNAKER_VERSION}" ]; then
        echo -e "${GREEN}✓ Found ${SPINNAKER_VERSION}${NC}"
        echo -e "${BLUE}Extracting and installing...${NC}"

        # Extract
        tar -xzf ${SPINNAKER_VERSION}
        SPINNAKER_DIR=$(basename ${SPINNAKER_VERSION} .tar.gz)
        cd ${SPINNAKER_DIR}

        # Install dependencies
        sudo apt-get install -y libusb-1.0-0

        # Install Spinnaker
        echo -e "${BLUE}Installing Spinnaker packages...${NC}"
        sudo dpkg -i libgentl*.deb
        sudo dpkg -i libspinnaker*.deb
        sudo dpkg -i libspinvideo*.deb
        sudo dpkg -i spinnaker*.deb
        sudo dpkg -i spinview*.deb

        # Configure USB permissions
        echo -e "${BLUE}Configuring USB permissions...${NC}"
        sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"1e10\", MODE=\"0666\"" > /etc/udev/rules.d/40-flir-spinnaker.rules'
        sudo udevadm control --reload-rules
        sudo udevadm trigger

        # Install PySpin (Python wrapper)
        echo -e "${BLUE}Installing PySpin (Python wrapper)...${NC}"
        if [ "$ARCH" = "aarch64" ]; then
            # Jetson ARM64
            pip3 install --user spinnaker_python*.whl || {
                echo -e "${YELLOW}⚠ PySpin wheel not found. Trying pip install...${NC}"
                pip3 install --user spinnaker-python
            }
        else
            # x86-64
            pip3 install --user spinnaker_python*.whl || {
                echo -e "${YELLOW}⚠ PySpin wheel not found. Trying pip install...${NC}"
                pip3 install --user spinnaker-python
            }
        fi

        cd ..

        echo -e "${GREEN}✓ Spinnaker SDK installed successfully!${NC}"
        echo -e "${GREEN}✓ You can now use FLIR Firefly cameras${NC}"

    else
        echo -e "${RED}✗ ${SPINNAKER_VERSION} not found in current directory${NC}"
        echo -e "${YELLOW}⚠ Skipping Spinnaker installation. UVC webcams will still work.${NC}"
        echo ""
        echo -e "${BLUE}To install Spinnaker later:${NC}"
        echo -e "  1. Download SDK from https://www.flir.com/products/spinnaker-sdk/"
        echo -e "  2. Run: ./install_linux.sh --with-firefly"
    fi
else
    echo -e "${BLUE}[6/6] Skipping FLIR Firefly support (use --with-firefly to enable)${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}✓ UVC webcam support: Ready${NC}"
if [ "$INSTALL_FIREFLY" = true ] && [ -f "${SPINNAKER_VERSION}" ]; then
    echo -e "${GREEN}✓ FLIR Firefly support: Installed${NC}"
else
    echo -e "${YELLOW}⚠ FLIR Firefly support: Not installed (use --with-firefly)${NC}"
fi
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Connect your RGB camera (UVC webcam or FLIR Firefly)"
echo -e "  2. Test camera: python3 camera_factory.py"
echo -e "  3. Run system: python3 main.py"
echo ""
echo -e "${BLUE}Supported cameras:${NC}"
echo -e "  • Generic UVC webcams (Logitech, Microsoft, etc.) - Works out-of-box"
echo -e "  • FLIR Firefly (global shutter) - Requires Spinnaker SDK"
echo ""
echo -e "${YELLOW}For troubleshooting, see README.md${NC}"
echo ""
