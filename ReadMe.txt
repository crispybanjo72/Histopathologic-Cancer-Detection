Metastatic Cancer Detection from Histopathology Patches

Project Overview
----------------
This project addresses the binary classification of histopathology image patches to detect metastatic cancer. The dataset is a modified version of the PatchCamelyon (PCam) benchmark, provided via Kaggle. Each image is a 96×96 RGB patch labeled as either tumor (1) or normal (0).

The objective is to build a robust deep learning model capable of identifying cancerous tissue, contributing to scalable diagnostic tools in digital pathology.

Repository Structure
--------------------
pcam-metastasis-detection/
├── notebooks/                  # EDA, model training, inference
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_inference_submission.ipynb
├── src/                        # Modular Python scripts
│   ├── model.py                # CNN, ResNet18, EfficientNetB0
│   ├── train.py                # Training loop with AUC tracking
│   ├── utils.py                # Dataset and dataloader setup
│   ├── inference.py            # Submission generation
├── submission/
│   ├── submission.csv          # Final Kaggle submission file
├── leaderboard_screenshot.png # Screenshot of leaderboard position
├── README.txt                  # Project overview and instructions
├── requirements.txt            # Python dependencies

Setup Instructions
------------------
1. Clone the repository:
   git clone https://github.com/your-username/pcam-metastasis-detection.git
   cd pcam-metastasis-detection

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare your data:
   - Place image patches in a folder (e.g., data/images/)
   - Ensure you have a CSV file with columns: id, label

4. Train the model:
   python src/train.py

5. Run inference:
   python src/inference.py

Model Architectures
-------------------
- Baseline CNN: Simple 2-layer convolutional network
- ResNet18: Pretrained residual network with modified output layer
- EfficientNetB0: Lightweight pretrained model optimized for small patches

Results
-------
- Evaluation metrics: AUC, accuracy, F1-score
- ROC curves and confusion matrices included in the notebook
- Best-performing model submitted to Kaggle

Kaggle Submission
-----------------
This project was submitted to the Kaggle competition.
Leaderboard screenshot is included in leaderboard_screenshot.png.

Author
------
Carson — Run Ralphie VII Run!