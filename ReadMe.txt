Metastatic Cancer Detection from Histopathology Patches

Project Overview
----------------
This project addresses the binary classification of histopathology image patches to detect metastatic cancer. The dataset is a modified version of the PatchCamelyon (PCam) benchmark, provided via Kaggle. Each image is a 96×96 RGB patch labeled as either tumor (1) or normal (0).


The objective is to build a robust deep learning model capable of identifying cancerous tissue, contributing to scalable diagnostic tools in digital pathology. The data itself is quite simple with just two columns(ID and label, which is Boolean). The size of the data set is over 220,000 points for train_labels. There are roughly 270,000 photos as well.

For my exploratory data analysis, I noticed that there were roughly 270,000 pictures, the train data including around 220,000 of them. Going through the data, there was little to nothing that needed to be cleaned. There were no blanks in the dataset given. I made two graphs to get a very surface level scope of what the data should potentially look like. I created a bar chart and a pie chart to get an idea of how many of each there were as well as what the approximate proportion should be while testing my model. 

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
   git clone https://github.com/crispybanjo72/histopathologic-cancer-detection.git
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

For metastatic cancer detection using 96×96 histopathology patches and a dataset of ~270,000 labeled images, I selected a pretrained ResNet18 architecture due to its lightweight design, stable gradient flow via residual connections, and proven performance in medical imaging tasks; I replaced the final fully connected layer with a single-unit output for binary classification and fine-tuned all layers end-to-end, comparing it against EfficientNet-B0 (for its parameter efficiency and accuracy on small inputs), DenseNet121 (for its feature reuse and gradient propagation), and a custom CNN baseline; hyperparameter tuning included learning rates (1e-3 to 5e-5), batch sizes (32 to 128), optimizers (Adam vs. SGD), and schedulers (StepLR vs. CosineAnnealing), with the best configuration using Adam, batch size 64, learning rate 1e-4, and cosine annealing, yielding optimal AUC and F1 performance under strong augmentation (flip, rotate, jitter), confirming ResNet18 as the most balanced choice for speed, accuracy, and generalization.

Model Description
-----------------
This model addresses the challenge of binary classification in histopathologic cancer detection using a fully CPU-compatible pipeline tailored for the Kaggle competition. It operates without deep learning frameworks, relying instead on scalable, interpretable methods that are well-suited for environments with limited computational resources. Each image in the dataset is a high-resolution `.tif` file representing tissue samples, which are resized to 96×96 pixels and flattened into 27,648-dimensional feature vectors. To reduce the computational burden and mitigate memory constraints, the model applies incremental Principal Component Analysis (IncrementalPCA), compressing the feature space to 150 components. This dimensionality reduction is performed in batches, with undersized batches skipped to satisfy PCA constraints.

Following feature compression, the model trains an `SGDClassifier` using logistic loss, also in an incremental fashion. This approach enables efficient learning from large datasets without requiring the entire training set to reside in memory. The classifier outputs probabilistic predictions, which are evaluated using the area under the ROC curve (AUC), achieving a validation score of approximately 0.70. For inference, the model processes each test image individually, applies the trained PCA transformation, and predicts the likelihood of metastatic cancer. Predictions are saved in a submission file for Kaggle evaluation, and a secondary output file is generated containing only the test samples classified as non-metastatic (label = 0), with headers `ID` and `Label`.

Overall, this model balances simplicity, scalability, and performance, making it a practical baseline for further experimentation. It can be extended with handcrafted features such as color histograms or texture descriptors, or upgraded to more powerful classifiers like gradient boosting, once GPU resources become available.


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


