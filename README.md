# YOLO Real-time Object Detection - Professional Edition

A professional-grade YOLO object detection system with GPU support, clean architecture, and comprehensive features.

## 🚀 Features

- **Multi-GPU Support**: Automatic CUDA detection with GPU selection
- **Clean Architecture**: Single-file modular design for easy maintenance
- **Real-time Processing**: Camera and video file detection
- **Batch Processing**: Process multiple images at once
- **Professional Logging**: Comprehensive logging system
- **Easy Configuration**: Simple command-line interface
- **Output Management**: Organized output directory structure

## 📁 Project Structure

```
yolo-detection/
├── yolo_detection.py            # Main detection application
├── run_quick.py                 # Quick test script
├── run_batch.py                 # Batch processing script
├── test_setup.py                # Setup verification script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── assets/                      # Asset files
│   ├── models/                  # YOLO model files
│   │   └── yolo11l.pt
│   └── videos/                  # Sample videos
│       └── test.mp4
├── output/                      # Output directory
├── logs/                        # Log files
└── docs/                        # Documentation
```

## 🛠️ Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. For GPU support (recommended)
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Verify installation
```bash
python test_setup.py
```

## 🎯 Usage

### Real-time Camera Detection
```bash
python yolo_detection.py
```

### Video File Detection
```bash
python yolo_detection.py --video assets/videos/test.mp4
```

### Quick Test
```bash
python run_quick.py
```

### Batch Processing
```bash
python run_batch.py input_folder output_folder
```

## ⚙️ Configuration Options

### Main Application (`yolo_detection.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | YOLO model path | `assets/models/yolo11l.pt` |
| `--device` | Device (cuda/cpu) | Auto-detect |
| `--gpu` | GPU ID (0, 1, ...) | `0` |
| `--confidence` | Confidence threshold | `0.5` |
| `--iou` | IoU threshold | `0.45` |
| `--camera` | Camera index | `0` |
| `--video` | Video file path | None (uses camera) |
| `--no-fps` | Hide FPS display | False |
| `--save-output` | Save output video | False |
| `--output-dir` | Output directory | `output/` |

### Examples

**Multi-GPU Selection:**
```bash
python yolo_detection.py --device cuda --gpu 0  # Use GPU 0
python yolo_detection.py --device cuda --gpu 1  # Use GPU 1
```

**Custom Settings:**
```bash
python yolo_detection.py \
    --video assets/videos/test.mp4 \
    --confidence 0.6 \
    --iou 0.5 \
    --save-output \
    --output-dir my_results/
```

**Batch Processing:**
```bash
python run_batch.py \
    input_images/ \
    output_results/ \
    --confidence 0.7 \
    --gpu 1
```

## 🎮 Controls

During detection:
- **'q'**: Quit application
- **'s'**: Save current frame
- **'space'**: Pause/resume (video mode only)

## 📊 Performance

With GPU acceleration:
- **Inference time**: ~20-35ms per frame
- **Real-time FPS**: 28-50 FPS
- **Multi-GPU support**: Choose optimal GPU

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Import Errors
```bash
# Run setup test
python test_setup.py

# Check dependencies
pip install -r requirements.txt
```

### Camera Issues
- Check camera permissions
- Try different camera index: `--camera 1`
- Ensure camera is not used by other applications

## 📝 Features

### Core Classes

#### `YOLODetector`
- GPU/CPU device management
- Model loading and inference
- Detection result processing

#### `VideoProcessor`
- Camera and video file processing
- Real-time FPS calculation
- Frame saving and output management

#### `DeviceManager`
- Multi-GPU detection and selection
- CUDA memory management
- Device information logging

### Scripts

#### `yolo_detection.py`
- Main application with full feature set
- Command-line interface
- Camera and video processing

#### `run_quick.py`
- Quick test with default settings
- Minimal configuration required

#### `run_batch.py`
- Batch processing for multiple images
- JSON output with detection results

#### `test_setup.py`
- Setup verification
- Dependency checking
- GPU testing

## 📄 License

This project is for educational and research purposes.

---

**Professional YOLO Detection System** - Clean, efficient, and production-ready.

# Hide FPS display
python real_time_detection.py --no-fps
```

### Command Line Options
- `--model`: YOLO model to use (default: yolo11l.pt)
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--confidence`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--camera`: Camera index (default: 0)
- `--no-fps`: Don't display FPS on screen

## Available Models

The script supports various YOLO11 models:
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large (default, good balance)
- `yolo11x.pt` - Extra Large (most accurate, slowest)

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame with detections
- **ESC**: Alternative quit method

## Performance Tips

### For Better Performance:
1. **Use GPU**: Ensure CUDA is installed and working
2. **Smaller Models**: Use yolo11s.pt or yolo11n.pt for faster inference
3. **Lower Resolution**: Reduce camera resolution in code if needed
4. **Adjust Thresholds**: Higher confidence threshold = fewer detections = faster processing

### GPU Requirements:
- NVIDIA GPU with CUDA support
- CUDA 11.0 or higher
- At least 4GB VRAM for YOLO11l

## Troubleshooting

### Common Issues:

1. **Camera Not Found**
   ```
   Error: Could not open camera 0
   ```
   - Check camera connection
   - Try different camera index (--camera 1, 2, etc.)
   - Ensure camera isn't used by another application

2. **CUDA Not Available**
   ```
   CUDA not available, using CPU
   ```
   - Install CUDA toolkit
   - Install PyTorch with CUDA support:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Model Download Issues**
   - First run downloads the model (500MB+ for yolo11l.pt)
   - Ensure internet connection
   - Models are cached after first download

4. **Low FPS**
   - Use smaller model (yolo11s.pt, yolo11n.pt)
   - Reduce camera resolution
   - Ensure GPU is being used
   - Close other applications

## System Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- Webcam or camera
- CPU with decent performance

### Recommended:
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.0+
- USB 3.0 camera for better performance

## File Structure

```
model_yolo/
├── real_time_detection.py  # Main detection script
├── test_setup.py          # Setup verification script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── detection_frame_*.jpg # Saved frames (created when 's' is pressed)
```

## Classes and Methods

### YOLODetector Class
- `__init__()`: Initialize detector with model and settings
- `detect_objects()`: Perform detection on single frame
- `run_camera_detection()`: Main detection loop with camera

## Performance Benchmarks

Approximate FPS on different hardware:
- **CPU Only**: 2-5 FPS
- **GTX 1060**: 15-25 FPS
- **RTX 3060**: 30-45 FPS
- **RTX 4080**: 60+ FPS

*Results may vary based on model size, resolution, and detection complexity*

## License

This project uses the Ultralytics YOLO implementation. Please refer to their licensing terms.

## Contributing

Feel free to submit issues or pull requests for improvements.
# vgmodel
