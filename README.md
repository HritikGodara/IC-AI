# 📸 CIFAR-10 Image Classifier: From Pixels to Predictions with Deep CNNs

This project walks through a complete deep learning workflow for multi-class image classification using the **CIFAR-10 dataset**. A custom **Convolutional Neural Network (CNN)** is built from scratch and trained using **TensorFlow** and **Keras** in a Jupyter Notebook environment. The model is trained, evaluated, and deployed for inference, showcasing best practices in data handling, architecture design, regularization, and training optimization.

---

## 📚 Table of Contents

- [📄 Project Summary](#-project-summary)  
- [✨ Key Features](#-key-features)  
- [🔁 Workflow & Methodology](#-workflow--methodology)  
- [📊 Results](#-results)  
- [🚀 How to Use](#-how-to-use)  
- [🔮 Future Improvements](#-future-improvements)

---

## 📄 Project Summary

The objective of this project is to develop a robust, from-scratch **Convolutional Neural Network** capable of classifying 32x32 color images into 10 distinct categories from the **CIFAR-10 dataset**. Through meticulous data preprocessing, model regularization, and training strategies, the model overcomes common pitfalls such as **overfitting** and **unstable gradients**.

The trained model is saved in the `.keras` format and is ready for inference or further fine-tuning.

---

## ✨ Key Features

- **Frameworks**: TensorFlow, Keras  
- **Dataset**: CIFAR-10 (60,000 32x32 color images in 10 classes)  
- **Model**: Custom-built Sequential CNN  
- **Saved Model**: `cifar10_model.keras`

### Techniques Implemented:
- 🔍 Data Normalization & Augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- 💡 Regularization: `Dropout`, `L2`
- 🧠 Training Optimization: `EarlyStopping`
- 📈 Evaluation Metrics: Accuracy, Precision, Recall

---

## 🔁 Workflow & Methodology

### 1. Environment Setup & Data Loading
- Developed and tested in **Kaggle Notebook** and **JupyterLab**.
- Images loaded via `image_dataset_from_directory`.

### 2. Data Preprocessing
- **Normalization**: Rescaled pixel values from [0, 255] → [0, 1].
- **Split**:
  - Training Set: 70%
  - Validation Set: 20%
  - Test Set: 10%

### 3. Model Architecture (Sequential CNN)
- 🧠 3× `Conv2D` Layers + ReLU Activation  
- 🧹 `MaxPooling2D` Layers for downsampling  
- 🚫 `Dropout` Layers to prevent overfitting  
- 🧾 `Flatten` → `Dense` (with L2 Regularization)  
- 🎯 Final `Dense` Output Layer with 10 units (Softmax)

### 4. Training Strategy
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Callback**: `EarlyStopping` (restores best weights based on validation loss)

### 5. Evaluation
- Performance tracked using training/validation plots.
- Evaluated on the test set using:
  - ✅ Accuracy
  - ✅ Precision
  - ✅ Recall

- Final inference tested on a custom external image.

---

## 📊 Results

The model demonstrated strong generalization with effective regularization:

- ✅ **Test Accuracy**: ~90% (Binary Accuracy)
- ✅ **Precision**: ~0.89

> **Note:** The use of Dropout + L2 Regularization + EarlyStopping effectively reduced overfitting seen in earlier training iterations.

---

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone [(https://github.com/HritikGodara/IC-AI)](https://github.com/HritikGodara/IC-AI)
cd IC-AI
````

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python matplotlib
```

### 3. Run the Notebook

Open `ic-ai.ipynb` in **Jupyter**, **VSCode**, or **Kaggle** and execute the cells sequentially.

### 4. Load the Pre-trained Model

```python
from tensorflow.keras.models import load_model

model = load_model('cifar10_model.keras')

# Example: Predict a single image
# Remember to preprocess (resize to 256x256 and normalize to [0, 1])
```

---

## 🔮 Future Improvements

* ⚙️ **Transfer Learning**
  Integrate pre-trained models like `ResNet50`, `VGG16`, or `EfficientNet` to achieve 90%+ top-1 accuracy.

* 🔍 **Hyperparameter Tuning**
  Use tools like `Keras Tuner` or `Optuna` to find the best values for learning rate, filter sizes, dropout, etc.

* 🧪 **Advanced Data Augmentation**
  Implement modern augmentation techniques like:

  * `CutMix`, `MixUp`
  * `Random Erasing`
  * ColorJitter and channel dropout

* 📊 **Class Imbalance Analysis**
  Investigate low recall scores and consider class-weighting or data rebalancing.

---

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Hritik G Bishnoi**
B.Tech CSE (AI/ML), 3rd Year | LPU
Aspiring AI Engineer | Passionate about Deep Learning & Computer Vision
---
