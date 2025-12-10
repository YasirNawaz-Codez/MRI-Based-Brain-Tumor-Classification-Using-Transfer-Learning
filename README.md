# 🧠 MRI-Based Brain Tumor Classification  
### Transfer Learning with VGG16, VGG19, ResNet50, MobileNetV2

This project implements MRI-based brain tumor classification using deep learning and transfer learning.  
The model classifies MRI images into **four categories**:
- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

---

## 📁 Dataset
Dataset from Kaggle: **Brain Tumor MRI Dataset**

- Total images: **7023**
- Training: **5712**
- Testing: **1311**
- Original size: **512×512 grayscale**

### Preprocessing
- Resized to **224×224**
- Normalized to **[0,1]**
- Grayscale → converted to **3 channels**
- Data Augmentation:
  - RandomFlip  
  - RandomRotation  
  - RandomZoom  

---

## 🚀 Models Used (Transfer Learning)
The following pre-trained CNN architectures were tested:

- **VGG16**
- **VGG19**
- **MobileNetV2**
- **ResNet50**
---

* **VGG16 and VGG19**
    VGG models are characterized by their extreme depth (16 and 19 layers) and use of small, uniform **$3 \times 3$ convolutional filters** stacked throughout the network. They established that depth is a key component for good performance, extracting features through a long sequence of basic operations. 

* **ResNet50 (Residual Networks)**
    ResNet introduced the concept of **"skip connections"** or **residual blocks** which allow the network to bypass layers and reuse activations from earlier layers. This design solves the vanishing gradient problem, enabling the training of much deeper networks (50 layers in this version) that perform well. 

* **MobileNetV2 (Efficient, low-latency architecture)**
    MobileNetV2 is designed for mobile and embedded vision applications where efficiency is critical. It uses **depthwise separable convolutions** and an inverted residual structure to drastically reduce the number of parameters and computational cost while maintaining high accuracy.
---
  
Each model was evaluated under different configurations:
- Optimizers: **Adam, SGD**
- Activations: **ReLU, GELU**
- Regularization: **Dropout, L2**
- Fine-tuning: **Last layers unfrozen**

---

## 🧪 Training Setup
- Epochs: **15**
- Batch Size: **32**
- Input Shape: **224×224×3**
- Callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

---

## 📈 Results Summary

| Model | Accuracy | Macro Recall | Macro F1 |
|-------|----------|--------------|----------|
| VGG16_Adam_L2 | 0.8383 | 0.8290 | 0.8278 |
| **VGG16_FineTune_LR1e-5** | **0.9169** | **0.9116** | **0.9119** |
| MobileNetV2_Adam_Drop0.4 | 0.8810 | 0.8746 | 0.8719 |
| ResNet50_Adam_L2 | 0.6308 | 0.6136 | 0.6067 |
| ResNet50_SGD_Optimizer | 0.4813 | 0.4577 | 0.4448 |
| ResNet50_GELU_Activation | 0.6484 | 0.6328 | 0.6296 |
| MobileNetV2_HighDrop0.6 | 0.8650 | 0.8605 | 0.8586 |

### 🏆 Best Model
**VGG16 Fine-Tuned (LR = 1e-5)**  

- Macro Recall: **0.9116**
- Accuracy: **91.69%**

---

## 📌 Key Observations
- Fine-tuning significantly improved VGG16 performance.
- MobileNetV2 gave strong results with less computation.
- ResNet50 underperformed in all tested configurations.
- Macro Recall is more important than accuracy for medical imaging.

---

## ▶️ How to Run

### Install Dependencies
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```


## 🗂️ Dataset Structure
Your dataset folder must follow this structure:

```text
Dataset/
├── Training/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No Tumor/
└── Testing/
    ├── Glioma/
    ├── Meningioma/
    ├── Pituitary/
    └── No Tumor/
```

## 3️⃣ Run Project
```bash
python mri_based_brain_tumor_classification.py
```

## 🧠 What I Learned

- Accuracy is not enough in medical imaging—models can have high accuracy and still miss tumors.
- Macro-averaged recall is the most reliable metric when every class (tumor type) is equally important.
- Fine-tuning improves performance significantly compared to using frozen feature extractors.
- Lightweight architectures like MobileNetV2 can perform surprisingly well on MRI data.
- Some CNNs like ResNet50 are highly sensitive to hyperparameters and might underperform without deeper tuning.
- The number of epochs directly affects convergence—many models showed improvement even at epoch 15.
- Proper preprocessing (resizing, normalization, channel expansion) is crucial when dealing with grayscale medical images.

---

## 🔗 References

### 📌 Dataset
**Kaggle — Brain Tumor MRI Dataset**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### 📌 Architecture & Deep Learning Resources (Geeksforgeeks)
- VGG Net Architecture Explained  
- MobileNet in Image Recognition  
- Residual Networks (ResNet)

### 📌 LLM Assistance
ChatGPT was used for:
- Research-related explanations  
- Understanding architecture differences  
- Clarifying concepts like epochs, fine-tuning, optimizers  
- Non-coding theoretical help  
