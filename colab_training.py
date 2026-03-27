"""
Google Colab Setup Script for Traffic Sign Recognition System
Run this in a Colab notebook cell by cell
"""

import os
import sys
import subprocess
from pathlib import Path

# ============================================================================
# STEP 1: GPU SETUP & VERIFICATION
# ============================================================================
def setup_gpu():
    """Verify and setup GPU in Colab"""
    print("=" * 60)
    print("STEP 1: GPU SETUP & VERIFICATION")
    print("=" * 60)
    
    # Check GPU availability
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f" GPU Name: {torch.cuda.get_device_name(0)}")
            print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f" CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU detected. Go to Runtime → Change runtime type → GPU")
    except Exception as e:
        print(f" Error checking GPU: {e}")
    print()


# ============================================================================
# STEP 2: MOUNT GOOGLE DRIVE (Optional)
# ============================================================================
def mount_drive():
    """Mount Google Drive to access/save files"""
    print("=" * 60)
    print("STEP 2: MOUNT GOOGLE DRIVE (Optional)")
    print("=" * 60)
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print(" Google Drive mounted at /content/drive")
        print("   Use this to store trained models and videos")
        return True
    except Exception as e:
        print(f" Could not mount Drive (not running in Colab): {e}")
        return False
    print()


# ============================================================================
# STEP 3: CLONE/SETUP PROJECT
# ============================================================================
def setup_project():
    """Clone or setup project in Colab"""
    print("=" * 60)
    print("STEP 3: PROJECT SETUP")
    print("=" * 60)
    
    project_dir = "/content/traffic_sign_system"
    
    # Option A: Clone from GitHub (if you have a repo)
    # Uncomment and modify if you push to GitHub
    # subprocess.run([
    #     "git", "clone",
    #     "YOUR_GITHUB_REPO_URL",
    #     project_dir
    # ], check=True)
    
    # Option B: Upload manually or use Drive
    if os.path.exists(project_dir):
        print(f" Project already exists at {project_dir}")
    else:
        print(f"Project directory: {project_dir}")
        print("   Upload your project files there or mount Drive")
    
    os.chdir(project_dir)
    print(f" Working directory: {os.getcwd()}")
    print()
    return project_dir


# ============================================================================
# STEP 4: INSTALL DEPENDENCIES
# ============================================================================
def install_dependencies():
    """Install required packages"""
    print("=" * 60)
    print("STEP 4: INSTALL DEPENDENCIES")
    print("=" * 60)
    
    packages = [
        "torch torchvision",
        "opencv-python",
        "pyyaml",
        "numpy",
        "ultralytics",  # YOLOv8
        "flask",        # Dashboard
        "pillow"
    ]
    
    for package in packages:
        print(f" Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
    
    print("All dependencies installed")
    print()


# ============================================================================
# STEP 5: DOWNLOAD DATASET
# ============================================================================
def download_dataset():
    """Download GTSRB dataset"""
    print("=" * 60)
    print("STEP 5: DOWNLOAD GTSRB DATASET")
    print("=" * 60)
    
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        print(" Downloading GTSRB dataset...")
        print("   (This may take 5-10 minutes, dataset is ~500MB)")
        
        from torchvision.datasets import GTSRB
        
        # Download training data
        print("   → Downloading training data...")
        GTSRB(data_dir, split='train', download=True)
        
        # Download test data
        print("   → Downloading test data...")
        GTSRB(data_dir, split='test', download=True)
        
        print(" Dataset downloaded successfully")
    except Exception as e:
        print(f" Error downloading dataset: {e}")
    print()


# ============================================================================
# STEP 6: TRAIN CNN CLASSIFIER
# ============================================================================
def train_cnn():
    """Train CNN classifier"""
    print("=" * 60)
    print("STEP 6: TRAIN CNN CLASSIFIER")
    print("=" * 60)
    
    try:
        print("🎓 Starting CNN training...")
        print("   This may take 20-30 minutes on GPU")
        
        subprocess.run([sys.executable, "train.py", "--mode", "cnn"], check=True)
        
        print("✅ CNN training completed")
        print("   Checkpoint saved to: checkpoints/cnn_classifier.pt")
    except Exception as e:
        print(f"❌ Error during training: {e}")
    print()


# ============================================================================
# STEP 7: TEST WITH SAMPLE VIDEO
# ============================================================================
def test_with_video():
    """Test with sample video or webcam"""
    print("=" * 60)
    print("STEP 7: TEST INFERENCE")
    print("=" * 60)
    
    print("📹 Options to test:")
    print("   1. Download sample video from internet")
    print("   2. Use video from Google Drive")
    print("   3. Use video uploaded to Colab")
    print()
    print("   For GPU optimized inference, use:")
    print("   >>> from modules.pipeline import RealtimePipeline")
    print("   >>> import yaml")
    print("   >>> cfg = yaml.safe_load(open('configs/config.yaml'))")
    print("   >>> cfg['inference']['frame_skip'] = 2")
    print("   >>> cfg['inference']['resize_factor'] = 0.75")
    print("   >>> pipeline = RealtimePipeline(cfg)")
    print()


# ============================================================================
# STEP 8: QUICK INFERENCE (Ready to use)
# ============================================================================
def setup_quick_inference():
    """Setup quick inference code"""
    print("=" * 60)
    print("STEP 8: QUICK INFERENCE SETUP")
    print("=" * 60)
    
    inference_code = '''
# Quick Inference Template (Copy-Paste)
# =====================================

import cv2
import yaml
from modules.pipeline import RealtimePipeline

# Load config
cfg = yaml.safe_load(open('configs/config.yaml'))

# OPTIMIZE FOR COLAB GPU
cfg['inference']['frame_skip'] = 1        # Process all frames (GPU is fast)
cfg['inference']['resize_factor'] = 0.75  # Reduce resolution slightly
cfg['yolo']['device'] = 'cuda'
cfg['cnn']['device'] = 'cuda'

# Initialize pipeline
pipeline = RealtimePipeline(cfg)

# Option A: Process video from Drive or Colab
# video_path = '/content/drive/MyDrive/sample_video.mp4'
# pipeline.run(source=video_path)

# Option B: Get single frame predictions
# frame = cv2.imread('path/to/image.jpg')
# annotated, detections, alerts = pipeline._process_frame(frame)
# cv2.imshow('Result', annotated)

print(" Inference ready! Modify video_path and run pipeline.run()")
'''
    
    print(inference_code)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all setup steps"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  GOOGLE COLAB SETUP - Traffic Sign Recognition System     ║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Run setup steps
    setup_gpu()
    mount_drive()
    setup_project()
    install_dependencies()
    download_dataset()
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  READY FOR TRAINING                                       ║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    response = input("Continue with CNN training? (y/n): ")
    if response.lower() == 'y':
        train_cnn()
    
    test_with_video()
    setup_quick_inference()


# ============================================================================
# COLAB NOTEBOOK CELLS (Copy-paste these into separate Colab cells)
# ============================================================================
"""
INSTRUCTIONS FOR GOOGLE COLAB:
==============================

CELL 1 - Setup and Install:
```python
%cd /content
!git clone YOUR_GITHUB_REPO_URL traffic_sign_system 2>/dev/null || echo "Repo already cloned"
%cd traffic_sign_system

# Run setup
exec(open('colab_training.py').read())
main()
```

OR manually run each cell:

CELL 2 - GPU Verification:
```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
```

CELL 3 - Install Dependencies:
```python
!pip install -q torch torchvision opencv-python pyyaml numpy ultralytics flask pillow
```

CELL 4 - Mount Drive (Optional):
```python
from google.colab import drive
drive.mount('/content/drive')
```

CELL 5 - Download Dataset:
```python
from torchvision.datasets import GTSRB
print("Downloading training data...")
GTSRB('data/raw', split='train', download=True)
print("Downloading test data...")
GTSRB('data/raw', split='test', download=True)
```

CELL 6 - Train CNN:
```python
!python train.py --mode cnn
```

CELL 7 - Run Inference:
```python
import cv2
import yaml
from modules.pipeline import RealtimePipeline

cfg = yaml.safe_load(open('configs/config.yaml'))
cfg['inference']['frame_skip'] = 1
cfg['inference']['resize_factor'] = 0.75
cfg['yolo']['device'] = 'cuda'
cfg['cnn']['device'] = 'cuda'

pipeline = RealtimePipeline(cfg)

# Option A: Process local video
!wget -q "https://example.com/sample_video.mp4" -O sample.mp4
pipeline.run(source='sample.mp4')

# Option B: Use video from Drive
# Modify video_path if stored in Drive
```

CELL 8 - Performance Test:
```python
import time
import yaml
from modules.pipeline import RealtimePipeline

cfg = yaml.safe_load(open('configs/config.yaml'))
cfg['yolo']['device'] = 'cuda'
cfg['cnn']['device'] = 'cuda'

pipeline = RealtimePipeline(cfg)

# Benchmark
print(" Speed Test (Colab GPU)")
start = time.time()
dummy_frames = 100
for i in range(dummy_frames):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _ = pipeline._process_frame(frame)
elapsed = time.time() - start

fps = dummy_frames / elapsed
print(f"Average FPS: {fps:.1f}")
print(f"Total time: {elapsed:.2f}s for {dummy_frames} frames")
```
"""

if __name__ == "__main__":
    main()
