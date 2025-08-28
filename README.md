Cat vs Dog Classifier Using MobileNetV2 + SVM

📌 Project Overview



This project implements a Cat vs Dog classifier using transfer learning with MobileNetV2 for feature extraction and Support Vector Machine (SVM) for classification.



The workflow includes:



Image loading \& preprocessing



Feature extraction using MobileNetV2 (pretrained on ImageNet)



Optional dimensionality reduction with PCA



Model training using Linear SVM



Evaluation and visualization of predictions



📊 Dataset



Source: Kaggle’s Cats vs Dogs dataset



Folder Structure:



datasets/

&nbsp;   train/

&nbsp;       cat.0.jpg

&nbsp;       dog.1.jpg

&nbsp;       ...





Total images: ~25,000



For CPU-based training, subset to ~500–1000 images per class



All images are placed directly inside the train folder, with filenames containing cat or dog to indicate the class.



🧮 Features Extracted



Feature extractor: MobileNetV2 pretrained on ImageNet



Input size: 64 × 64 RGB images



Features per image: ~1280 (before PCA)



Target labels:



0 → Cat 🐱



1 → Dog 🐶



⚙️ Requirements



Python 3.x



Libraries: numpy, matplotlib, scikit-learn, tensorflow



Install via:



pip install -r requirements.txt



🛠 Methodology

1️⃣ Data Loading



Load images from the train folder



Determine class from filename (cat or dog)



Limit number of samples per class for memory efficiency



2️⃣ Preprocessing



Resize all images to 64×64



Normalize pixel values to \[0, 1]



Create NumPy arrays for features and labels



3️⃣ Feature Extraction



Use MobileNetV2 (imagenet weights, exclude top layers)



Extract high-level feature embeddings



Optional: Apply PCA for dimensionality reduction



4️⃣ Model Training



Train a Linear SVM on extracted features



Optionally use LinearSVC for faster CPU training



5️⃣ Model Evaluation



Evaluate on test set



Compute accuracy score



Visualize sample predictions (correct ✅ / incorrect ❌)



📈 Results



Accuracy (with ~1000–2000 samples): 80–90%



Accuracy improves with larger dataset and GPU training



Visualization:



Correct predictions labeled ✅



Misclassifications labeled ❌



🔮 Future Improvements



Increase dataset size for better generalization



Apply data augmentation to reduce overfitting



Fine-tune MobileNetV2 layers (instead of using fixed features)



Experiment with advanced classifiers (Random Forest, XGBoost, CNN-based models)



Deploy the model using Flask or Streamlit



👤 Author



Ankita Das

Machine Learning Intern @ Skillcraft Technology

