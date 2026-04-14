<div align="center">

# 🌿 Medicinal Leaf Disease Detection with AI-MedLeafX + XAI

<p align="center">
  Deep learning for medicinal plant leaf disease classification with <b>ResNet</b>, <b>EfficientNetV2-S</b>, <b>ConvNeXt-Tiny</b>, and explainability using <b>Grad-CAM</b> & <b>t-SNE</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-AI--MedLeafX-2E8B57" />
  <img src="https://img.shields.io/badge/Classes-13-blue" />
  <img src="https://img.shields.io/badge/XAI-GradCAM%20%7C%20t--SNE-purple" />
  <img src="https://img.shields.io/badge/Best%20Accuracy-98.16%25-success" />
</p>

</div>

---

## 📌 Overview

This project focuses on **medicinal plant leaf disease classification** using deep learning on the **AI-MedLeafX** dataset. In addition to classification performance, the project also emphasizes **model interpretability** through Explainable AI (XAI) techniques such as **Grad-CAM** and **t-SNE**.

### Main objectives

- Classify diseases on medicinal plant leaves from RGB images.
- Compare multiple CNN architectures on the same dataset.
- Analyze model decisions using XAI.
- Build a solid baseline for future real-world agricultural applications.

---

## ✨ Highlights

- ✅ Fine-tuning pretrained CNN models on **AI-MedLeafX**
- ✅ Comparison of **ResNet50**, **EfficientNetV2-S**, and **ConvNeXt-Tiny**
- ✅ Explainability with **Grad-CAM** and **t-SNE**
- ✅ Strong performance with **98.85% accuracy** from **EfficientNetV2-S**
- ✅ Organized notebooks for training, evaluation, and XAI experiments

---

## 🖼️ Project Preview

### Sample leaf images

<p align="center">
  <img src="results/__results___files/output1.png" width="30%" />
  <img src="results/__results___files/12.png" width="30%" />
  <img src="results/__results___files/324.png" width="30%" />
</p>
  <p align="center">
  <img src="results/__results___files/output3.png" width="30%" />
  <img src="results/__results___files/3123.png" width="30%" />
  <img src="results/__results___files/1343.png" width="30%" />
</p>

### Example result visualizations

<p align="center">
  <img src="results/__results___files/__results___12_0.png" width="90%" />
  <img src="results/__results___files/__results___12_1.png" width="90%" />
</p>

<p align="center">
<img src="results/__results___files/__results___12_4.png" width="90%" /></p>

## 🎯 Problem Statement

Early detection of plant diseases is important for reducing crop damage and improving decision-making in smart agriculture. Unlike many previous works that focus on common agricultural crops, this project targets **medicinal plants**, which are less studied and have more limited public datasets.

The task is to classify leaf images into disease/healthy categories using deep learning models and provide visual explanations for model predictions.

---

## 🧪 Dataset

The project uses **AI-MedLeafX**, a medicinal plant disease dataset containing **4 plant species** and **13 classes**.

### Plant species

- Camphor
- HariTaki
- Neem
- Sojina

### Disease categories

- Bacterial Spot
- Shot Hole
- Powdery Mildew
- Yellow Leaf
- Healthy Leaf

### Input setting

- Image size: **224 × 224**
- Data format: RGB images
- Preprocessing based on **ImageNet normalization**

### Split used in notebook

The notebook uses the augmented image directory and splits the dataset as follows:

- **Train:** 70%
- **Validation:** 20%
- **Test:** 10%

### Observed dataset size in notebook

- **65,178 images**
- **13 classes**
- Train: **45,624**
- Validation: **13,035**
- Test: **6,519**

> Note: The original report mentions the base AI-MedLeafX dataset at around **10,858 images**. The notebook appears to use an augmented version, resulting in a much larger number of samples.

---

## 🧠 Models

Three representative CNN architectures were studied:

### 1. ResNet50

A strong baseline CNN with residual connections that helps stabilize deep training.

### 2. EfficientNetV2-S

A more compute-efficient architecture balancing model size and classification performance.

### 3. ConvNeXt-Tiny

A modern ConvNet inspired by design ideas from Vision Transformers, achieving the best performance in this project.

---

## ⚙️ Methodology

The overall pipeline follows these main steps:

```text
Input Images
   ↓
Preprocessing & Augmentation
   ↓
Train / Validation / Test Split
   ↓
Fine-tune Pretrained CNN Models
   ↓
Evaluation Metrics
   ↓
XAI Analysis (Grad-CAM, t-SNE)
```

### Preprocessing

Implemented using `torchvision.transforms`:

- Resize to `224x224`
- Random horizontal flip
- Random rotation
- Convert to tensor
- Normalize with ImageNet mean/std

### Training strategy

Based on the report:

- Pretrained ImageNet weights
- AdamW optimizer
- Cross-Entropy Loss
- Early stopping
- ReduceLROnPlateau
- Two-stage training strategy for stable convergence

### Evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 📊 Results

### Quantitative comparison

| Model            |    Accuracy (%) |      Precision |         Recall |       F1-score | Params |
| ---------------- | --------------: | -------------: | -------------: | -------------: | -----: |
| ResNet50         |           95.03 |           0.95 |           0.95 |           0.95 |   ~24M |
| EfficientNetV2-S | **98.85** | **0.99** | **0.99** | **0.99** |   ~21M |
| ConvNeXt-Tiny    |           98.16 |           0.98 |           0.98 |           0.98 |   ~28M |

### Key observations

- **ConvNeXt-Tiny** achieved the best overall performance.
- **EfficientNetV2-S** provided a strong balance between accuracy and efficiency.
- **ResNet50** remained a reliable baseline for comparison.

---

## 🔍 Explainable AI

This project does not stop at prediction performance. It also investigates **why** the model makes a prediction.

### Grad-CAM

Grad-CAM helps identify the image regions that most influence the final prediction.

Findings from the report:

- Good localization on diseases such as **Bacterial Spot** and **Shot Hole**.
- Distributed attention over the leaf surface for **Healthy Leaf** samples.
- More difficulty with **Powdery Mildew**, where symptoms are diffuse and spread across the leaf.

### t-SNE

`t-SNE` is used to visualize learned feature embeddings in 2D.

Observations:

- **ResNet50** forms reasonably separated clusters but still shows overlap.
- **EfficientNetV2-S** produces the clearest feature separation.
- **ConvNeXt-Tiny** forms denser clusters yet still achieves the highest classification accuracy.

---

## 🚀 Installation

Recommended environment: **Python 3.10+**

### Install dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow opencv-python tqdm jupyter lime grad-cam kaggle
```

### Kaggle API setup

Place your Kaggle API file at:

```bash
~/.kaggle/kaggle.json
```

Set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## ⚡ Quick Start

### 1. Launch Jupyter

```bash
jupyter notebook
```

### 2. Open the main training notebook

```text
medleaf-disease-detection (2).ipynb
```

### 3. Download dataset from Kaggle

Inside the notebook, a command similar to this is used:

```bash
kaggle datasets download -d mrlocbap/ai-medleafx
```

### 4. Extract data

The notebook extracts the archive into:

```text
content/ai-medleafx
```

### 5. Train and evaluate

Main notebook tasks include:

- loading images by class directory
- creating train/val/test splits
- fine-tuning pretrained models
- saving checkpoints
- evaluating classification results
- visualizing XAI outputs

---

## 🖼 Inference on New Images

The `test.ipynb` notebook includes helper functions for:

- single-image prediction
- Grad-CAM visualization
- LIME explanation
- loading trained `.pth` weights for inference

Typical inference flow:

1. Load model checkpoint
2. Apply the same test transform
3. Run forward pass
4. Return predicted class and confidence score

This can be extended into:

- a simple desktop app
- a Streamlit/Gradio web app
- a mobile app
- a field-support diagnostic tool

---

## ⚠️ Limitations

Despite strong results, several limitations remain:

- The training data is mostly collected in controlled conditions, not fully reflecting field environments.
- Diffuse diseases such as powdery mildew remain harder to localize and explain.
- The system has not yet been deployed as a real-time production application.

---

## 🔮 Future Work

Possible next steps include:

- training on real-world field images for better generalization;
- experimenting with **ViT**, **Swin Transformer**, or hybrid CNN-Transformer models;
- using additional XAI tools such as **SHAP** and deeper LIME analysis;
- building a more user-friendly diagnosis interface;
- deploying the model on mobile devices or agricultural drones.

---

## 🙏 Acknowledgement

- **Supervisor:** TS. Lê Thị Vĩnh Thanh
- **Course:** Thị giác máy tính và ứng dụng
- **Institution:** Trường Đại học Công nghiệp TP.HCM

---

## 📄 Internal Sources

This README was consolidated from:

- `Nhom_01_NhanDienBenhTrenLaCay.docx`
- `medleaf-disease-detection (2).ipynb`
- `test.ipynb`

---

## ⭐ Suggested Next Improvements

If you want to make this repository even more professional, the next best additions would be:

- a clean architecture diagram;
- confusion matrix images with labels;
- Grad-CAM comparison figure per model;
- a `requirements.txt` file;
- a `demo.ipynb` or `app.py` for quick demonstration.
