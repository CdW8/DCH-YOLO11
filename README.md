# DCH-YOLO11

DCH-YOLO11 is an enhanced object detection model based on YOLO11, specifically designed for accurate recognition of multi-symptom citrus Huanglongbing (HLB) leaves under complex field conditions. This project introduces three innovative modules to improve the original YOLO11 architecture, boosting feature extraction, fine-grained recognition, and generalization across challenging scenarios.

## Key Features

- **Three Innovative Modules:**
  - `C3k2_DFF` (Dynamic Feature Fusion): Strengthens interaction between global and local features for better detection of subtle early-stage symptoms.
  - `C2PSA_CAA` (Context Anchor Attention): Focuses on complex leaf vein regions and enhances the model’s ability to distinguish diverse symptoms.
  - `HDFPN` (High-Efficiency Dynamic Feature Pyramid Network): Optimizes multi-scale feature fusion for robust detection across object sizes.
- **Model Architecture**  
  ![DCH-YOLO11 Architecture](./Fig/DCH-YOLO11.png)
- **Dependencies**  
  All required packages are listed in `requirements.txt` for easy setup.

## Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/CdW8/DCH-YOLO11.git
    cd DCH-YOLO11
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Training & Inference**
    - Please refer to the code and comments in the `ultralytics` folder for training and inference instructions.

## Application Scenarios

- Multi-symptom HLB leaf detection
- Plant disease identification in complex field environments
- Other fine-grained object detection tasks

## Citation

If you find this project useful, please consider citing or starring the repository!

---
