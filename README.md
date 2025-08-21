Cat vs Dog Classifier Using MobileNetV2 + SVM
📌 Project Overview

This repository implements a Cat vs Dog classifier using transfer learning with MobileNetV2 for feature extraction and Support Vector Machine (SVM) for classification.
The project covers:

Image loading & preprocessing

Feature extraction with MobileNetV2 (pretrained on ImageNet)

Dimensionality reduction with PCA (optional)

Model training with Linear SVM

Model evaluation & visualization of predictions

📊 Dataset

Source: Kaggle’s Cats vs Dogs dataset

Structure:

datasets/
│── train/          ← training set (used for fitting the model)
│   ├── cats/       ← cat images for training
│   └── dogs/       ← dog images for training
│
│── test/           ← test set (used for evaluating the model)
    ├── cats/       ← cat images for testing
    └── dogs/       ← dog images for testing

Sample Size: 25,000 images (can be reduced for CPU training, e.g., 500–1000 per class).

🧮 Features Extracted

We use MobileNetV2 (pretrained on ImageNet) to generate deep feature vectors for each image.
These extracted features are then used to train the SVM classifier.

Input size: 64 × 64 RGB images

Extracted features per image: ~1280 (before PCA)

Target labels:

0 → Cat 🐱

1 → Dog 🐶

⚙️ Requirements

Make sure you have Python 3.x installed along with the following libraries:

numpy
matplotlib
scikit-learn
tensorflow


Install via:

pip install -r requirements.txt

🛠 Methodology
1️⃣ Data Loading

Loaded images from train/cats and train/dogs.

Limited number of samples per class (to avoid memory crashes).

2️⃣ Preprocessing

Resized all images to 64×64.

Normalized pixel values to [0, 1].

Created NumPy arrays for features (images) and labels (labels).

3️⃣ Feature Extraction

Used MobileNetV2 (imagenet weights, exclude top).

Extracted high-level feature embeddings.

Optionally applied PCA to reduce dimensionality.

4️⃣ Model Training

Trained a Linear SVM (Support Vector Machine) on the extracted features.

Compared with LinearSVC for speed optimization on CPU.

5️⃣ Model Evaluation

Evaluated on test set.

Calculated accuracy score.

Visualized sample predictions with labels (Cat 🐱 / Dog 🐶).

📥 Installation & Usage

Clone this repository:

git clone https://github.com/yourusername/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier


Place dataset inside datasets/ directory.

Open the notebook:

jupyter notebook cat_vs_dog_classifier.ipynb


Run the notebook sequentially to reproduce:

Data preprocessing

Feature extraction

Model training

Prediction & visualization

📈 Results

Accuracy (with ~1000–2000 samples): 80–90%

Accuracy improves with larger dataset & GPU training

Visualization:

Correct predictions labeled ✅

Misclassifications labeled ❌

📊 Features

Preprocessing: Image resizing, normalization

Feature Extraction: MobileNetV2 embeddings

Model Training: Linear SVM

Evaluation: Accuracy score, sample predictions visualization

🔮 Future Improvements

Increase dataset size for better generalization

Apply data augmentation to reduce overfitting

Fine-tune MobileNetV2 layers (instead of fixed features)

Try advanced classifiers (Random Forest, XGBoost, CNN-based classifier)

Deploy model via Flask / Streamlit

👤 Author

Ankita Das
Machine Learning Intern @ Skillcraft Technology