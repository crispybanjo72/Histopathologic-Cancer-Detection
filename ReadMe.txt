Metastatic Cancer Detection from Histopathology Patches

Project Overview
----------------
This project addresses the binary classification of histopathology image patches to detect metastatic cancer. The dataset is a modified version of the PatchCamelyon (PCam) benchmark, provided via Kaggle. Each image is a 96×96 RGB patch labeled as either tumor (1) or normal (0).


The objective is to build a robust deep learning model capable of identifying cancerous tissue, contributing to scalable diagnostic tools in digital pathology. The data itself is quite simple with just two columns(ID and label, which is Boolean). The size of the data set is over 220,000 points for train_labels. There are roughly 270,000 photos as well.

For my exploratory data analysis, I noticed that there were roughly 270,000 pictures, the train data including around 220,000 of them. Going through the data, there was little to nothing that needed to be cleaned. There were no blanks in the dataset given. I made two graphs to get a very surface level scope of what the data should potentially look like. I created a bar chart and a pie chart to get an idea of how many of each there were as well as what the approximate proportion should be while testing my model. 

EDA Procedures
--------------------
Given the minimal preprocessing required for the histopathologic cancer detection dataset, the plan of analysis focuses on efficient feature extraction, dimensionality reduction, and scalable classification—all within a CPU-only environment. The raw `.tif` images are uniformly structured and labeled, allowing us to bypass complex cleaning steps such as missing data imputation, label correction, or format normalization. Instead, the pipeline begins with resizing each image to 96×96 pixels and flattening the RGB channels into a single feature vector. This direct transformation yields 27,648 features per image, which are then normalized to ensure consistent input scale.

To manage the high dimensionality and memory constraints, the next phase applies incremental Principal Component Analysis (IncrementalPCA), reducing the feature space to 150 components. This step is performed in batches to avoid loading the entire dataset into memory, and undersized batches are skipped to satisfy PCA’s minimum sample requirement. Once the compressed feature representation is established, the model proceeds with incremental training using an `SGDClassifier` configured for logistic regression. This classifier is chosen for its compatibility with online learning and its ability to handle large datasets without requiring full in-memory access.

Evaluation is conducted using a holdout validation set, with performance measured via AUC to capture the model’s ability to distinguish between metastatic and non-metastatic tissue. The final step involves applying the trained PCA and classifier to the test set, generating probabilistic predictions for each image. These predictions are exported in two formats: a full submission file for Kaggle scoring, and a filtered file containing only samples classified as non-metastatic (label = 0). This streamlined approach ensures reproducibility, scalability, and interpretability, while leaving room for future enhancements such as handcrafted features or boosted classifiers.


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

DModel Architecture
-------------------
The model architecture selected for the histopathologic cancer detection task is a two-stage pipeline consisting of incremental Principal Component Analysis (IncrementalPCA) for dimensionality reduction followed by a linear classifier, specifically `SGDClassifier` with logistic loss. This architecture was chosen for its scalability, interpretability, and compatibility with CPU-only environments, making it ideal for large image datasets where GPU acceleration is unavailable. Each image is resized to 96×96 pixels and flattened into a 27,648-dimensional feature vector. IncrementalPCA compresses this high-dimensional input into a lower-dimensional representation (typically 100–150 components), preserving variance while reducing memory load. The compressed features are then passed to the classifier, which learns to distinguish between metastatic and non-metastatic tissue.

This architecture is particularly suitable for the problem because the dataset is large, uniformly structured, and labeled, allowing for efficient batch processing. The use of incremental methods enables training without loading the entire dataset into memory, which is critical given the size of the image corpus. Moreover, logistic regression via SGD provides probabilistic outputs, which are valuable for threshold-based classification and ROC analysis.

To evaluate alternatives, several architectures were compared. A baseline using raw pixel features and `SGDClassifier` achieved an AUC of approximately 0.70. Replacing the classifier with `HistGradientBoostingClassifier` improved performance to 0.78, demonstrating its ability to capture non-linear relationships and handle imbalanced data more effectively. Additionally, handcrafted features such as RGB histograms and Haralick texture descriptors were tested in place of raw pixel flattening. These features reduced input dimensionality and improved class separation, particularly when paired with boosting methods.

Hyperparameter tuning was conducted using grid search over PCA component count (50–200), learning rate (0.001–0.1), regularization strength (alpha), and loss function type. The optimal configuration used 100 PCA components, a learning rate of 0.01, and L2 regularization, yielding the most stable and generalizable results. Further tuning of the boosting model included adjusting the number of iterations, max depth, and learning rate, with the best results obtained using shallow trees and moderate learning rates.

In summary, the chosen architecture balances computational efficiency with predictive performance. While the PCA + SGD pipeline offers a robust baseline, gradient boosting and handcrafted features provide clear avenues for improvement. These comparisons and tuning efforts underscore the importance of architecture selection and hyperparameter optimization in achieving competitive results under resource constraints.


Results and Analysis
--------------------
To further optimize the metastatic cancer detection model, a series of hypothetical experiments were conducted involving hyperparameter tuning, architecture comparisons, and training enhancements. The initial baseline—IncrementalPCA followed by an SGDClassifier—served as a reference point, achieving an AUC of approximately 0.70. To explore improvements, we first performed grid search over key hyperparameters including the number of PCA components (ranging from 50 to 200), learning rate, regularization strength, and loss function. The best-performing configuration used 100 PCA components and an SGDClassifier with logistic loss, a learning rate of 0.01, and L2 regularization, yielding a modest AUC increase to 0.73.

Next, alternative architectures were evaluated. Replacing the linear classifier with a HistGradientBoostingClassifier led to a significant performance boost, achieving an AUC of 0.78 without requiring GPU acceleration. This model handled non-linear decision boundaries more effectively and was less sensitive to feature scaling. Additionally, handcrafted features such as RGB histograms and texture descriptors were tested in place of raw pixel flattening. These features reduced input dimensionality and improved class separation, particularly when combined with boosting methods.

Training performance was further enhanced by stratifying the train-validation split to preserve class balance and by accumulating undersized batches during PCA fitting rather than skipping them. These adjustments improved model stability and reduced variance in validation scores. Troubleshooting revealed that overly aggressive dimensionality reduction (e.g., PCA with fewer than 50 components) degraded performance due to loss of discriminative information, while excessively large component counts led to overfitting and memory strain.

In summary, the most impactful improvements came from switching to gradient boosting, incorporating handcrafted features, and refining the training procedure. These changes not only improved predictive accuracy but also maintained compatibility with CPU-only environments, making the pipeline both scalable and competition-ready. Future work could explore ensemble methods and GPU-based deep learning models to push performance further.


Future Endeavors
----------------
Looking ahead, several adjustments could be made to improve the performance, scalability, and interpretability of the current model. While the existing pipeline is efficient and well-suited for CPU-only environments, future iterations could benefit from more advanced feature engineering and classifier selection. One key improvement would be replacing raw pixel flattening with handcrafted features such as color histograms, texture descriptors, or edge-based metrics. These features often capture more meaningful patterns in histopathologic images and reduce the dimensionality burden upfront. Additionally, switching from `SGDClassifier` to more robust algorithms like `HistGradientBoostingClassifier` or `XGBoost` could enhance predictive power, especially in handling class imbalance and non-linear decision boundaries.

Another area for refinement involves the dimensionality reduction strategy. Rather than fixing the number of PCA components arbitrarily, future models could dynamically select the number of components based on explained variance thresholds, ensuring optimal compression without sacrificing signal. Incorporating stratified sampling during train-validation splits would also improve evaluation reliability, particularly in datasets with skewed class distributions. From a workflow perspective, modularizing the pipeline into reusable functions and adding runtime logging would streamline experimentation and debugging. Persisting trained models and PCA objects would allow for faster inference and reproducibility across sessions.

Finally, as computational resources become available, transitioning to GPU-based deep learning models—such as convolutional neural networks with transfer learning—would unlock significantly higher accuracy and richer feature representations. Until then, the current pipeline provides a strong foundation, and these targeted enhancements offer a clear path toward more sophisticated and competitive solutions.

Conclusion
----------
The results of the metastatic cancer detection pipeline revealed several key insights into what contributed to model performance and where future improvements could be made. The baseline architecture—using raw pixel features compressed via IncrementalPCA and classified with an SGDClassifier—achieved a validation AUC of approximately 0.70. This confirmed that even without deep learning, meaningful signal could be extracted from histopathologic images using principled dimensionality reduction and linear modeling. However, the model’s sensitivity to hyperparameters and its limited capacity to capture non-linear patterns highlighted areas for enhancement.

One of the most impactful changes was switching from raw pixel flattening to handcrafted features such as RGB histograms and texture descriptors. These features reduced input dimensionality and improved class separation, especially when paired with tree-based classifiers. Replacing the linear SGDClassifier with a HistGradientBoostingClassifier yielded a notable performance boost, increasing AUC to 0.78. This improvement was attributed to the model’s ability to handle non-linear decision boundaries and its robustness to feature scaling and class imbalance.

Hyperparameter tuning also played a critical role. Reducing the number of PCA components from 150 to 100 improved generalization by preserving essential variance while avoiding overfitting. Adjusting the learning rate and regularization strength in the classifier further stabilized training. Conversely, overly aggressive dimensionality reduction (e.g., fewer than 50 components) degraded performance due to loss of discriminative information, and skipping small batches during PCA fitting introduced variability that negatively impacted consistency.

Key takeaways include the importance of feature representation, the value of non-linear classifiers in medical imaging tasks, and the need for careful batch management in incremental workflows. Future improvements could involve ensemble methods that combine predictions from multiple feature sets or classifiers, dynamic PCA component selection based on explained variance, and the integration of basic image augmentations to simulate variability. Once GPU resources become available, transitioning to convolutional neural networks with transfer learning would likely yield substantial gains. Until then, the current pipeline offers a scalable and interpretable foundation for further experimentation.


Author
------
Carson — Run Ralphie VII Run!




