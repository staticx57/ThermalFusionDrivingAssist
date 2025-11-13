# FLIR Boson Thermal Road Monitor

Real-time object detection and driver alert system using FLIR Boson thermal camera, optimized for NVIDIA Jetson Orin with GPU acceleration.

## Features

- **GPU-Accelerated Detection**: YOLOv8 with TensorRT optimization for maximum performance
- **Thermal Imaging**: Works with FLIR Boson thermal cameras (320x256 or 640x512)
- **Real-Time Alerts**: Visual and prioritized alerts for road hazards
- **Smart Object Detection**: Focuses on road-relevant objects (vehicles, pedestrians, animals, signs)
- **Performance Monitoring**: Live GPU, CPU, memory, and thermal metrics
- **Driver-Friendly GUI**: Clear, high-contrast display with alert panels and metrics

## Hardware Requirements

- **NVIDIA Jetson Orin** (Nano, NX, or AGX)
- **FLIR Boson Thermal Camera** (connected via USB or CSI)
- Minimum 4GB RAM (8GB+ recommended)

## Software Requirements

- JetPack 5.0+ (Ubuntu 20.04/22.04)
- Python 3.8+
- CUDA 11.4+ (included with JetPack)
- TensorRT 8.5+ (included with JetPack)

## Installation

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopencv-dev python3-opencv
```

### 2. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Install Jetson Stats (Recommended)

```bash
sudo pip3 install jetson-stats
```

### 4. Download YOLO Model (First Time)

The application will automatically download the YOLOv8 model on first run. You can also pre-download:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Model Options:**
- `yolov8n.pt` - Nano (fastest, ~6MB)
- `yolov8s.pt` - Small (balanced, ~22MB)
- `yolov8m.pt` - Medium (more accurate, ~52MB)

## Usage

### Basic Usage

```bash
python3 main.py
```

### With Options

```bash
# Use specific camera device
python3 main.py --camera-id 0

# Use larger model for better accuracy
python3 main.py --model yolov8s.pt

# Adjust confidence threshold
python3 main.py --confidence 0.6

# Start in fullscreen
python3 main.py --fullscreen

# For Boson 320x256
python3 main.py --width 320 --height 256
```

### Full Command Line Options

```
--camera-id INT       Camera device ID (default: 0)
--width INT           Camera frame width (default: 640)
--height INT          Camera frame height (default: 512)
--model PATH          YOLO model path (default: yolov8n.pt)
--confidence FLOAT    Detection threshold 0.0-1.0 (default: 0.5)
--use-tensorrt        Use TensorRT optimization (default: True)
--no-tensorrt         Disable TensorRT optimization
--gpu-id INT          GPU device ID (default: 0)
--fullscreen          Start in fullscreen mode
```

## Keyboard Controls

- **Q** or **ESC**: Quit application
- **F**: Toggle fullscreen
- **D**: Toggle detection visualization
- **S**: Save screenshot
- **P**: Print performance summary to console

## Camera Setup

### USB Connection

1. Connect FLIR Boson via USB adapter
2. Check device: `ls /dev/video*`
3. Use device number with `--camera-id`

### CSI Connection

1. Connect to CSI port on Jetson
2. May require camera driver configuration
3. Check available devices: `v4l2-ctl --list-devices`

## Performance Optimization

### TensorRT Conversion

On first run with `--use-tensorrt`, the YOLO model will be converted to TensorRT engine format. This takes 5-10 minutes but is cached for future runs.

### Power Mode

For maximum performance:

```bash
# Set to maximum power mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Expected Performance

- **Jetson Orin Nano**: 30-50 FPS with yolov8n + TensorRT
- **Jetson Orin NX**: 50-80 FPS with yolov8n + TensorRT
- **Jetson Orin AGX**: 80-120 FPS with yolov8n + TensorRT

## Alert System

### Alert Levels

1. **CRITICAL** (Red): Object very close, immediate attention required
2. **WARNING** (Orange): Object approaching, prepare to react
3. **INFO** (Cyan): Object detected, situational awareness

### Detection Categories

- **Vehicles**: Cars, trucks, buses, motorcycles
- **Vulnerable Road Users**: Pedestrians, bicycles
- **Animals**: Dogs, cats, birds (thermal signature detection)
- **Traffic Control**: Stop signs, traffic lights

## Troubleshooting

### Camera Not Detected

```bash
# Check video devices
ls -l /dev/video*

# Test camera
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### Low FPS

1. Ensure TensorRT is enabled: `--use-tensorrt`
2. Use smaller model: `--model yolov8n.pt`
3. Set max power mode: `sudo jetson_clocks`
4. Check thermal throttling: Press **P** in app

### CUDA/TensorRT Errors

```bash
# Verify CUDA installation
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch for Jetson
pip3 install --upgrade torch torchvision
```

### Jtop Not Working

```bash
# Reinstall jetson-stats
sudo pip3 uninstall jetson-stats
sudo pip3 install jetson-stats
sudo reboot
```

## Architecture

```
┌─────────────────┐
│  FLIR Boson     │ Thermal camera input
└────────┬────────┘
         │
┌────────▼────────┐
│  flir_camera.py │ Camera interface (V4L2)
└────────┬────────┘
         │
┌────────▼────────┐
│object_detector  │ YOLOv8 + TensorRT on GPU
└────────┬────────┘
         │
┌────────▼────────┐
│ road_analyzer   │ Alert generation logic
└────────┬────────┘
         │
┌────────▼────────┐
│  driver_gui     │ Real-time display
└─────────────────┘
```

## License

This project is provided as-is for educational and research purposes.

## Safety Notice

⚠️ **This system is designed as a driver assistance tool only. It does NOT replace human judgment and attention. Always maintain full control of your vehicle and follow all traffic laws.**