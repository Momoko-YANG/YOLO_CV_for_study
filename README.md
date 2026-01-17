# Deep Learning Based Gesture Recognition System

> ⚠️ **Disclaimer**: This is a **personal learning project**, intended only for learning and researching the application of YOLO object detection algorithms in gesture recognition. The code and documentation are for reference only and not recommended for production use.

A real-time gesture recognition system built on YOLOv5/YOLOv8 models with a user-friendly graphical interface. The system can recognize gestures from images, videos, and real-time camera input.

## Project Overview

This project aims to explore the application of deep learning in computer vision, particularly the implementation of YOLO model framework for gesture recognition tasks. The system adopts a modular design, combining Python and GUI interface technology to achieve a one-stop gesture recognition application architecture.

### Learning Objectives
- Understand the principles and implementation of YOLO series object detection algorithms
- Master the use of PyTorch deep learning framework
- Learn to build desktop GUI applications using PySide6
- Understand the application of computer vision in real-world scenarios

## Features

### Core Features
- **Multi-model Support**: Supports both YOLOv5n and YOLOv8 models with real-time model switching
- **Multiple Input Sources**:
  - Static image recognition
  - Video file processing  
  - Real-time camera detection
- **Gesture Category Recognition**: Recognizes three main gesture types
  - Stop (open palm)
  - NumberTwo (victory gesture)
  - Understand (fist)

### System Features
- **Real-time Performance**: Detection time ~0.03 seconds, supports 30 FPS processing
- **Interactive GUI**: Intuitive user interface built with PySide6
- **User Management**: SQLite-based login/registration system
- **Result Logging**: Comprehensive detection result recording with confidence scores and timestamps
- **Heatmap Visualization**: Visualization of model attention regions
- **Parameter Adjustment**: Real-time adjustment of confidence and IOU thresholds

## System Requirements

### Hardware Requirements
- **Processor**: Intel Core i9-13900HX or equivalent
- **GPU**: NVIDIA GeForce RTX 4060 or higher (8GB VRAM recommended)
- **Memory**: Minimum 16GB
- **Camera**: 0.9MP or higher resolution (for real-time detection)

### Software Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python Version**: 3.10.14 or higher

## Dependencies

```txt
torch==2.0.1
torchvision
ultralytics==8.1.3
opencv-python==4.7.0.72
PySide6
numpy
Pillow
matplotlib
sqlite3
```

## Installation

### 1. Clone the Project
```bash
git clone <repository-url>
cd gesture-recognition-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
- Place YOLOv5n model weights in `weights/yolov5n.pt`
- Place YOLOv8 model weights in `weights/yolov8.pt`

### 5. Run the Application
```bash
python scripts/run_main_login.py
```

## Project Structure

```
gesture-recognition-system/
├── datasets/                       # Dataset directory
│   └── (gesture datasets)
├── icons/                          # Icon resources
├── runs/                           # Training results
│   └── (model training outputs)
├── test_media/                     # Test media files
│   └── (test images, videos)
├── themes/                         # UI theme styles
├── ultralytics/                    # Ultralytics YOLO core library
├── weights/                        # Model weight files
│   └── (pretrained models)
│
├── src/                            # Source code
│   ├── LoginForm.py / .ui          # Login form
│   ├── LoginWindow.py              # Login window logic
│   ├── Recognition_UI.py / .ui     # Recognition system UI
│   ├── Recognition_UI_ui.py        # Compiled UI code
│   ├── RecSystem.py / .qrc         # Main system module
│   ├── RecSystem_rc.py             # Resource compiled file
│   ├── System_login.py             # Main program with login
│   ├── System_noLogin.py           # Main program without login
│   └── YOLOv8v5Model.py            # YOLO model wrapper
│
├── scripts/                        # Run and test scripts
│   ├── run_main_login.py           # Entry point (with login)
│   ├── run_main_noLogin.py         # Entry point (without login)
│   ├── run_test_camera.py          # Camera test script
│   ├── run_test_image.py           # Image test script
│   ├── run_test_video.py           # Video test script
│   ├── run_train_model.py          # Model training script
│   └── test.py                     # Test script
│
├── output_examples/                # Output examples
│   ├── video_*.avi                 # Detection result video
│   └── table_data_*.csv            # Detection data table
│
├── __init__.py                     # Python package init
├── requirements.txt                # Project dependencies
├── environment.txt                 # Environment setup guide
└── README.md                       # Project documentation
```

## Usage

### Starting the Application
1. Run `python scripts/run_main_login.py`
2. Register a new account or login with existing credentials
3. Select detection model (YOLOv5n or YOLOv8)

### Detection Modes

#### Image Detection
- Click "Load Image" button
- Select image file (jpg, png, etc.)
- View detection results with bounding boxes and confidence scores

#### Video Detection
- Click "Load Video" button
- Select video file (mp4, avi, etc.)
- Frame-by-frame processing with real-time visualization

#### Camera Detection
- Click "Start Camera" button
- Perform real-time gesture recognition
- Adjust CONF and IOU thresholds as needed

## Model Performance

| Model | Precision | Recall | mAP@0.5 | Detection Speed |
|-------|-----------|--------|---------|-----------------|
| YOLOv8 | 95%+ | 95%+ | 0.963 | ~0.03s |
| YOLOv5n | 94%+ | 93%+ | 0.940 | ~0.03s |

### Training Results Analysis
- **Loss Convergence**: Box loss, classification loss, and object loss all converge stably
- **PR Curve Performance**: mAP@0.5 reaches 0.963, showing excellent detection performance
- **Confusion Matrix**: Recognition accuracy for all categories is around 95%

## Dataset Information

- **Data Source**: Based on HaGRID open-source dataset
- **Total Images**: 11,886
  - Training set: 10,953 images
  - Validation set: 604 images
  - Test set: 329 images
- **Image Resolution**: 640×640 pixels
- **Class Distribution**: 3 gesture categories, evenly distributed

### Data Preprocessing
- Automatic orientation correction, removing EXIF orientation info
- Unified image size to 640×640 pixels
- Data augmentation: mosaic augmentation, rotation, scaling, etc.

## Configuration

### Adjustable Parameters
- **CONF Threshold**: Minimum confidence score for detection (default: 0.5)
- **IOU Threshold**: Non-maximum suppression threshold (default: 0.5)
- **Frame Rate**: Processing speed for video/camera input (locked at 30 FPS)

### Model Switching
- Supports real-time switching between YOLOv5n and YOLOv8
- Supports uploading custom trained weight files

## System Interface

### Login/Registration Page
- SQLite database stores user information
- Supports user registration, login, and password modification
- CAPTCHA mechanism to prevent malicious registration

### Main Interface
- Left control panel: Input source selection, model switching
- Right display area: Original image preview, heatmap display
- Bottom data table: Detection result recording and statistics
- Top parameter adjustment: Real-time CONF and IOU threshold adjustment

## Results and Logging

The system provides comprehensive logging functionality:
- Detection timestamps
- Confidence scores
- Bounding box coordinates
- Gesture classification results
- Support for result export for analysis

## Technical Details

### System Architecture
- **Detection Framework**: YOLO (You Only Look Once)
- **Deep Learning Library**: PyTorch 2.0.1
- **Computer Vision**: OpenCV 4.7.0.72
- **GUI Framework**: PySide6
- **Database**: SQLite3

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: Dynamic learning rate with warmup phase
- **Batch Size**: Hardware-adaptive
- **Data Augmentation**: Mosaic, rotation, scaling, etc.
- **Loss Function**: Multi-component loss (box, objectness, classification)

### Core Algorithms
#### YOLO Detection Pipeline
1. Image preprocessing: resize, normalization
2. Feature extraction: CNN backbone network
3. Object detection: Regression of bounding boxes and classification probabilities
4. Post-processing: Non-maximum suppression (NMS)

#### Gesture Recognition Pipeline
1. Input source reading (image/video/camera)
2. Data preprocessing
3. Model inference
4. Result post-processing
5. Visualization display

## Known Limitations

- Performance may vary under different lighting conditions
- Recognition accuracy depends on hand visibility and gesture clarity
- Current implementation limited to three gesture categories
- Complex backgrounds may affect detection accuracy

## Future Improvements

### Feature Extensions
- **More Gesture Categories**: Extend support for more gesture types
- **Cross-cultural Adaptation**: Research gesture differences across cultures
- **Mobile Deployment**: Optimize models for mobile device deployment
- **Multi-modal Input**: Integrate voice, text, and other information modalities

### Technical Optimization
- **Model Lightweighting**: Optimize models based on Neural Architecture Search (NAS)
- **Real-time Enhancement**: Further optimize inference speed
- **Robustness Enhancement**: Improve recognition stability in complex environments

## License

This project is for personal learning use only. Please do not use for commercial purposes.

## Acknowledgments

- **Dataset**: [HaGRID](https://github.com/hukenovs/hagrid) Open-source Gesture Dataset
- **Technical Framework**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), PyTorch, OpenCV Open Source Community

---

**Note**: This is a personal learning project. Code quality and performance may have limitations and is for reference only.

## Appendix

### Troubleshooting

**Q: Model loading failed**
A: Check if the model file path is correct and ensure the model file is complete

**Q: Camera won't start**  
A: Check camera permission settings and ensure the camera is not occupied by other programs

**Q: Detection accuracy is poor**
A: Adjust CONF and IOU thresholds, ensure good lighting conditions

**Q: System runs slowly**
A: Check if GPU drivers are correctly installed, consider using a lighter model

### Performance Optimization Tips

1. **Hardware Optimization**: Use CUDA-supported NVIDIA GPU
2. **Environment Configuration**: Ensure PyTorch correctly recognizes GPU
3. **Parameter Tuning**: Adjust model parameters according to actual needs
4. **Memory Management**: Clean up unused variables and cache promptly

### Extension Development Guide

This system adopts modular design for easy feature extension:

- **Add Gesture Categories**: Modify dataset and model training configuration
- **Interface Customization**: UI design based on PySide6 framework
- **Algorithm Replacement**: Implement YOLOv8v5Detector interface to integrate new algorithms
- **Database Extension**: Extend more user features based on SQLite
