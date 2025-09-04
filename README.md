Cat vs Dog Classifier — MobileNetV2 + SVM

📌 Project Overview



This project implements a Cat vs Dog image classifier by combining:



Transfer Learning – MobileNetV2 (pretrained on ImageNet) for robust feature extraction



Support Vector Machine (SVM) – linear kernel for final classification



This approach avoids training a deep network from scratch and delivers strong accuracy even on CPUs.



📊 Dataset



Source: Kaggle Cats vs Dogs



Structure (after extraction):



datasets/

&nbsp; train/

&nbsp;   cat.0.jpg

&nbsp;   dog.1.jpg

&nbsp;   ...





~25,000 total images (balanced)



For quick CPU experiments, subset to ~500–1000 per class



Filenames contain cat or dog → automatically used as labels



🧮 Features Extracted

Item	Value

Extractor	MobileNetV2 (ImageNet weights, include\_top=False)

Input Size	160 × 160 RGB

Embedding Dim.	1280

Labels	0 → Cat 🐱, 1 → Dog 🐶

⚙️ Requirements

pip install numpy matplotlib seaborn scikit-learn tensorflow



🛠 Methodology



1️⃣ Data Loading



Loads all .jpg / .png files from the train folder



Derives class from filename



2️⃣ Preprocessing



Resize to 160×160



MobileNetV2 preprocess\_input normalization



3️⃣ Feature Extraction



MobileNetV2 (frozen, avg-pooling head)



Produces 1280-D embeddings



4️⃣ Model Training



Linear SVM trained on extracted features



train\_test\_split (80/20, stratified)



5️⃣ Evaluation \& Visualization



Accuracy, confusion matrix, classification report



Random sample predictions rendered with true/pred labels



📈 Results

Setting	#Images (per class)	Accuracy

CPU, 500 each	1000	~96%

CPU, full dataset	25,000	97–98%



Sample Confusion Matrix



&nbsp;	Pred Cat	Pred Dog

True Cat	96	4

True Dog	3	97

🔮 Future Improvements



Add data augmentation for better generalization



Fine-tune upper MobileNetV2 layers on Cats vs Dogs



Try RBF kernel SVM or Logistic Regression baseline



Deploy via Streamlit / Flask



Package as Docker container for reproducibility



👤 Author



Ankita Das

Machine Learning Intern — Skillcraft Technology

