# Thermal Fusion Driving Assist - Enhanced Features

## Overview

This project combines FLIR Boson thermal imaging with RGB cameras for enhanced driver assistance. The system features multi-modal sensor fusion, smart proximity alerts, and cross-platform support for both Jetson Orin (ARM) and x86-64 workstations.

## New Features (v2.0)

### 🔥 RGB Camera Integration
- ✅ USB webcam support via V4L2
- ✅ Jetson CSI camera support via GStreamer
- ✅ Auto-detection of available RGB cameras
- ✅ Synchronized thermal + RGB capture

### 🎨 Multi-View Display Modes
- **Thermal Only** - Classic thermal imaging view
- **RGB Only** - Standard RGB camera view
- **Fusion** - Thermal + RGB combined (multiple algorithms)
- **Side-by-Side** - Thermal and RGB displayed horizontally
- **Picture-in-Picture** - Thermal inset in RGB (or vice versa)

Switch views with:
- GUI button: `VIEW: [mode]`
- Keyboard: Press `V` to cycle through modes

### 🌈 Advanced Fusion Algorithms

1. **Alpha Blend** - Weighted average of thermal + RGB
   - Adjustable blend ratio (α = 0.3, 0.5, 0.7)
   - Good for general-purpose fusion

2. **Edge Enhanced** - RGB base with thermal edge overlay
   - Preserves RGB detail
   - Highlights thermal boundaries in red

3. **Thermal Overlay** - Hot regions overlaid on RGB
   - Shows only thermal hotspots (top 30% intensity)
   - Clear RGB base with thermal highlights

4. **Side-by-Side** - Dual view comparison
   - Thermal and RGB displayed together
   - Best for analysis and debugging

5. **Picture-in-Picture** - Compact dual view
   - One camera inset in the other
   - Saves screen space

6. **Max Intensity** - Maximum pixel values from both
   - Preserves brightest features
   - Good for high-contrast scenes

7. **Feature Weighted** - Adaptive blending
   - Uses thermal where edges are strong
   - Uses RGB where texture is rich
   - Best overall image quality

### 🚨 Smart Proximity Alerts (NEW!)

**Red Pulse Warnings** - Visual driver assistance system:

- **LEFT SIDE ALERT** - Pulsing red bar on left when pedestrian/animal detected on left
- **RIGHT SIDE ALERT** - Pulsing red bar on right when pedestrian/animal detected on right
- **CENTER WARNING** - Top banner with "COLLISION WARNING" for objects ahead

Features:
- 2 Hz pulsing effect (smooth sine wave)
- Intensity varies with object priority:
  - **CRITICAL** (red) - Pedestrians, bicycles, motorcycles
  - **WARNING** (orange) - Other moving objects
- Displays object count in each zone
- Non-intrusive overlay (doesn't block video)

### 🖥️ Enhanced GUI Layout

**Old Layout Problems:**
- ❌ Cramped buttons (7 buttons in 1 row at 75% scale)
- ❌ Only shows 2 alerts
- ❌ Overlays block video
- ❌ No multi-view support

**New Layout Features:**
- ✅ 2-row button layout (larger, easier to click)
- ✅ Shows up to 4 alerts (increased from 2)
- ✅ Better button organization by function
- ✅ Fusion controls (mode + alpha slider)
- ✅ View mode indicator (top-left)
- ✅ Smart proximity alerts (sides + top)

**Button Layout:**

```
Row 1: PAL: [ironbow] | YOLO: ON | BOX: ON | DEV: CUDA | MODEL: V8N
Row 2: FLUSH: OFF | SKIP: 1/1 | VIEW: FUSION | FUS: ALPHA | α: 0.5
```

### 🔧 Camera Calibration System

The fusion system includes a calibration utility to align RGB and thermal cameras:

**Three Calibration Methods:**

1. **Checkerboard Pattern** (Most Accurate)
   ```bash
   python3 camera_calibration.py
   # Select method 1, show checkerboard to both cameras
   ```

2. **Feature Matching** (Automatic)
   - Uses ORB feature detection
   - Works with natural scenes
   - Good for outdoor calibration

3. **Manual Point Selection** (Most Flexible)
   - Click 4+ corresponding points
   - Works in any environment
   - Good when automatic methods fail

**Calibration Output:**
- Saves to `camera_calibration.json`
- Homography matrix for RGB → Thermal alignment
- Used automatically by fusion processor

### 🖥️ Cross-Platform Support

**Jetson Orin (ARM + CUDA):**
- VPI GPU acceleration ✅
- CSI camera support (GStreamer) ✅
- CUDA acceleration ✅
- Optimized for deployment

**x86-64 Workstation (ThinkPad P16):**
- USB camera support ✅
- CPU/CUDA detection fallback ✅
- OpenCV-based processing ✅
- Ideal for development/debugging

**Platform Detection:**
```python
# Automatically detects:
- Architecture (ARM vs x86-64)
- Jetson presence (/etc/nv_tegra_release)
- VPI availability
- Graceful fallback to CPU if needed
```

## Installation

### Prerequisites

**Both Platforms:**
```bash
pip3 install opencv-python numpy ultralytics torch torchvision psutil Pillow
```

**Jetson-Specific:**
```bash
# VPI (pre-installed on Jetson)
sudo apt-get install nvidia-vpi2

# jetson-stats (optional, for GPU monitoring)
sudo -H pip3 install jetson-stats
```

### Quick Start

**1. Thermal Only (Original Mode):**
```bash
python3 main_vpi.py --detection-mode model --model yolov8n.pt
```

**2. Thermal + RGB Fusion (New):**
```bash
python3 main_fusion.py --detection-mode model --model yolov8n.pt
```

**3. With Calibration:**
```bash
# First, calibrate cameras
python3 camera_calibration.py

# Then run with calibration
python3 main_fusion.py --calibration-file camera_calibration.json
```

**4. Thermal Only (Disable RGB):**
```bash
python3 main_fusion.py --disable-rgb
```

## Command-Line Options

```
usage: main_fusion.py [-h] [--camera-id ID] [--width W] [--height H]
                      [--disable-rgb] [--confidence C] [--detection-mode {edge,model}]
                      [--model PATH] [--device {cuda,cpu}] [--fullscreen]
                      [--scale SCALE] [--palette {white_hot,black_hot,...}]
                      [--fusion-mode {alpha_blend,edge_enhanced,...}]
                      [--fusion-alpha ALPHA] [--calibration-file PATH]

Camera Options:
  --camera-id ID        Thermal camera device ID (auto-detect if omitted)
  --disable-rgb         Disable RGB camera (thermal-only mode)
  --width W             Camera frame width (default: 640)
  --height H            Camera frame height (default: 512)

Detection Options:
  --confidence C        Detection threshold (default: 0.25)
  --detection-mode      'model' for YOLO, 'edge' for edge detection
  --model PATH          YOLO model: yolov8n.pt, yolov8s.pt, yolov8m.pt
  --device              'cuda' for GPU, 'cpu' for CPU-only

Display Options:
  --fullscreen          Start in fullscreen mode
  --scale SCALE         Display scaling (default: 2.0)
  --palette             Thermal colormap (default: ironbow)

Fusion Options:
  --fusion-mode         Fusion algorithm (default: alpha_blend)
  --fusion-alpha        Blend ratio: 0.0=RGB, 1.0=Thermal (default: 0.5)
  --calibration-file    Path to calibration JSON file
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| `F` | Toggle fullscreen |
| `V` | Cycle view modes (Thermal → RGB → Fusion → ...) |
| `Y` | Toggle YOLO detection on/off |
| `D` | Toggle detection boxes |
| `C` | Cycle thermal palettes |
| `G` | Toggle device (CUDA ↔ CPU) |
| `S` | Save screenshot |
| `P` | Print performance stats |
| `1-8` | Select palette directly |

## GUI Button Controls

**Row 1 - Detection Controls:**
- `PAL: [name]` - Cycle thermal palettes
- `YOLO: ON/OFF` - Toggle object detection
- `BOX: ON/OFF` - Toggle detection box overlay
- `DEV: CUDA/CPU` - Toggle GPU/CPU processing
- `MODEL: V8N/V8S/V8M` - Cycle YOLO models

**Row 2 - Performance & View:**
- `FLUSH: ON/OFF` - Toggle buffer flushing (lower latency)
- `SKIP: 1/N` - Frame skip for detection (1/1, 1/2, 1/3, 1/4)
- `VIEW: [mode]` - Cycle view modes
- `FUS: [mode]` - Cycle fusion algorithms (if RGB available)
- `α: X.X` - Adjust fusion blend ratio (if RGB available)

## File Structure

```
ThermalFusionDrivingAssist/
├── main_vpi.py                 # Original thermal-only entry point
├── main_fusion.py              # NEW: Thermal + RGB fusion entry point
│
├── flir_camera.py              # FLIR Boson thermal camera interface
├── rgb_camera.py               # NEW: RGB camera interface (USB/CSI)
├── camera_detector.py          # Camera auto-detection
│
├── vpi_detector.py             # VPI-accelerated detection
├── object_detector.py          # YOLO object detector
├── road_analyzer.py            # Alert generation logic
│
├── fusion_processor.py         # NEW: Thermal-RGB fusion algorithms
├── camera_calibration.py       # NEW: Camera alignment calibration
│
├── driver_gui.py               # Original GUI
├── driver_gui_v2.py            # NEW: Enhanced GUI with multi-view
│
├── performance_monitor.py      # System metrics monitoring
├── requirements.txt            # Python dependencies
└── *.sh                        # Deployment scripts
```

## Fusion Mode Comparison

| Mode | Use Case | Pros | Cons |
|------|----------|------|------|
| **Alpha Blend** | General purpose | Balanced, adjustable | Can look washed out |
| **Edge Enhanced** | Detail preservation | Sharp RGB + thermal edges | Edges may be noisy |
| **Thermal Overlay** | Hot object detection | Clear RGB base | Only shows hotspots |
| **Side-by-Side** | Analysis/debugging | See both sources | Takes more space |
| **Picture-in-Picture** | Space-saving | Compact, both views | Small inset |
| **Max Intensity** | High contrast | Preserves bright features | Can amplify noise |
| **Feature Weighted** | Best quality | Adaptive, optimal blend | Slower processing |

**Recommendations:**
- **Daytime:** Edge Enhanced or Alpha Blend (α=0.3)
- **Nighttime:** Thermal Overlay or Alpha Blend (α=0.7)
- **Analysis:** Side-by-Side
- **Driving:** Feature Weighted or Alpha Blend

## Smart Proximity Alert System

The smart alert system divides the field of view into three zones:

```
┌─────────────────────────────────────────────┐
│           ⚠️ OBJECT AHEAD ⚠️                │  ← Center Warning
├────────┬────────────────────────┬───────────┤
│        │                        │           │
│  LEFT  │       CENTER           │   RIGHT   │
│  ZONE  │       ZONE             │   ZONE    │
│   !!   │                        │    !!     │
│        │                        │           │
│  (n)   │                        │   (n)     │
│        │                        │           │
└────────┴────────────────────────┴───────────┘
   ↑                                    ↑
Red pulse on side              Red pulse on side
when objects detected         when objects detected
```

**Alert Levels:**
- **Critical Objects** (person, bicycle, motorcycle) → Bright red pulse (30-80% opacity)
- **Warning Objects** (motion, animals) → Orange pulse (20-50% opacity)
- **Pulse Frequency:** 2 Hz (smooth sine wave)

## Camera Calibration Workflow

For accurate fusion, cameras should be calibrated:

**Method 1: Checkerboard (Recommended)**
```bash
# 1. Print a checkerboard pattern (9x6 or 7x5 internal corners)
# 2. Run calibration wizard
python3 camera_calibration.py

# 3. Select method 1
# 4. Position checkerboard visible to BOTH cameras
# 5. Follow prompts
# 6. Calibration saved to camera_calibration.json
```

**Method 2: Manual (Quick)**
```bash
# 1. Run calibration wizard
python3 camera_calibration.py

# 2. Select method 3 (manual)
# 3. Click 4-6 corresponding points in both views
#    (corners, edges, distinctive features)
# 4. Calibration saved
```

**Using Calibration:**
```bash
python3 main_fusion.py --calibration-file camera_calibration.json
```

## Performance Optimization

**For Jetson Orin Nano (8GB):**
```bash
# Best performance
python3 main_fusion.py \
  --model yolov8n.pt \
  --device cuda \
  --fusion-mode alpha_blend \
  --scale 1.5

# Enable frame skipping for smoother GUI
# Click SKIP button to cycle: 1/1 → 1/2 → 1/3 → 1/4
```

**For x86-64 Workstation:**
```bash
# Development/debugging
python3 main_fusion.py \
  --model yolov8s.pt \
  --device cuda \
  --fusion-mode feature_weighted \
  --scale 2.0
```

## Troubleshooting

### RGB Camera Not Detected

```bash
# Test RGB camera detection
python3 rgb_camera.py

# List available cameras
v4l2-ctl --list-devices
```

### VPI Not Available (x86-64)

```
Warning: VPI not available on this platform
```
This is **normal** on x86-64. The system automatically falls back to CPU processing.

### Fusion Performance Issues

- Try simpler fusion mode: `--fusion-mode alpha_blend`
- Reduce resolution: `--width 320 --height 256`
- Enable frame skip: Click `SKIP` button
- Use lighter YOLO model: `--model yolov8n.pt`

### Calibration Failed

- Ensure both cameras see the same scene
- Use good lighting
- For checkerboard: print at least 8"x10" size
- For manual: choose distinctive features (corners, edges)

## Development Workflow

**On Workstation (x86-64 ThinkPad P16):**
```bash
# Develop and test with USB cameras
python3 main_fusion.py --device cuda

# Test CPU fallback
python3 main_fusion.py --device cpu
```

**Deploy to Jetson Orin:**
```bash
# Copy files to Jetson
scp -r * jetson@192.168.1.100:~/ThermalFusionDrivingAssist/

# SSH to Jetson
ssh jetson@192.168.1.100

# Run on Jetson with CSI camera
cd ~/ThermalFusionDrivingAssist
python3 main_fusion.py --device cuda
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    THERMAL CAMERA                           │
│                    (FLIR Boson 640x512)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ Grayscale thermal data
                      ↓
            ┌──────────────────────┐
            │  Thermal Palette     │ → Colorized thermal
            │  (ironbow, lava...)  │
            └──────────┬───────────┘
                       │
                       ↓
        ┌──────────────────────────────────┐
        │                                  │
        ↓                                  ↓
┌────────────────┐              ┌──────────────────────┐
│  RGB CAMERA    │              │   FUSION PROCESSOR   │
│  (USB/CSI)     │──────────→   │  • Alpha Blend       │
│  640x480       │ RGB frame    │  • Edge Enhanced     │
└────────────────┘              │  • Thermal Overlay   │
                                │  • Feature Weighted  │
                                └──────────┬───────────┘
                                           │ Fused frame
                                           ↓
                                ┌──────────────────────┐
                                │   VIEW SELECTOR      │
                                │  Thermal/RGB/Fusion  │
                                └──────────┬───────────┘
                                           │
                                           ↓
                                ┌──────────────────────┐
                                │  YOLO DETECTOR       │
                                │  (VPI accelerated)   │
                                └──────────┬───────────┘
                                           │ Detections
                                           ↓
                                ┌──────────────────────┐
                                │  ROAD ANALYZER       │
                                │  Generate alerts     │
                                └──────────┬───────────┘
                                           │ Alerts
                                           ↓
                        ┌──────────────────────────────────┐
                        │  ENHANCED GUI                    │
                        │  • Multi-view display            │
                        │  • Smart proximity alerts        │
                        │  • Red pulse warnings            │
                        │  • Detection boxes               │
                        └──────────────────────────────────┘
```

## Future Enhancements

- [ ] 3D depth estimation from thermal-RGB fusion
- [ ] Lane detection and departure warnings
- [ ] Recording/playback of fused video
- [ ] Network streaming (RTSP/WebRTC)
- [ ] Machine learning model training on fused data
- [ ] Multi-camera support (360° coverage)
- [ ] Integration with vehicle CAN bus

## License

See LICENSE file for details.

## Credits

- FLIR Boson SDK
- NVIDIA VPI (Vision Programming Interface)
- Ultralytics YOLOv8
- OpenCV community

---

**For questions or issues, please open a GitHub issue.**
