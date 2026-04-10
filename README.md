# 🩺 Chest X-Ray Classification using Ensemble Learning

A deep learning project for **automated classification of chest X-ray images** into:

* COVID-19
* Pneumonia
* Normal

The system uses **ensemble learning techniques** to improve prediction accuracy, reduce overfitting, and enhance generalization compared to individual CNN models.

---

## 🚀 Overview

Medical image classification is difficult due to:

* High similarity between COVID-19 and Pneumonia
* Limited dataset size
* Overfitting in deep learning models

This project solves these issues using:

* Multiple CNN models
* Ensemble learning (Bagging / Voting)
* Data preprocessing and augmentation

---

## 🧠 Methodology

### 🔹 Data Preprocessing

* Image resizing
* Normalization
* Data augmentation (rotation, flipping)

### 🔹 Model Training

* CNN-based models (Transfer Learning)
* Optimizer: Adam
* Loss Function: Cross-Entropy

### 🔹 Ensemble Learning

* Predictions from multiple models are combined
* Final prediction is based on aggregated outputs

👉 This improves robustness and reduces variance

---

## 📂 Project Structure

```id="q1c0q0"
project/
│
├── data/                # Dataset (not included)
├── outputs/             # Results, graphs
├── utils/               # Helper functions
│
├── main.py              # Training pipeline
├── predict.py           # Inference script
├── generate_results.py  # Evaluation metrics
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash id="db3b8c"
git clone https://github.com/AnshKene/medical-xray-ensemble.git
cd medical-xray-ensemble
```

### 2. Create Virtual Environment

```bash id="pyr9i3"
python -m venv myenv
myenv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash id="a6n1db"
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train Model

```bash id="u2yzvd"
python main.py
```

### 🔹 Generate Evaluation Results

```bash id="y1ks8x"
python generate_results.py
```

### 🔹 Predict on New Image

```bash id="twx7rf"
python predict.py
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📈 Expected Results

* Improved accuracy using ensemble learning
* Reduced overfitting compared to single models
* Better generalization on unseen data

---

## 🧪 Dataset

* Total Images: 1125
* Classes:

  * COVID-19
  * Pneumonia
  * Normal

⚠️ Dataset is not included in the repository.

---

## 🔍 Key Features

✔ Ensemble Learning
✔ Transfer Learning
✔ Data Augmentation
✔ Modular Scripts (train / predict / evaluate)

---

## ⚠️ Limitations

* Small dataset size
* Performance depends on data quality
* No real-world deployment yet

---

## 🔮 Future Work

* Implement **Stacking Ensemble (advanced)**
* Add **Grad-CAM visualization**
* Deploy using **Flask / FastAPI**
* Optimize for lightweight systems

---

## 👨‍💻 Contributors

* Ansh Kene
* Vyas Thakre

---

## 📜 License

For academic and research purposes only.
