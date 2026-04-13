<div align="center">

# 🌿 Medicinal Leaf Disease Detection with AI-MedLeafX + XAI

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

# English Version

## 📌 Overview
This project focuses on **medicinal plant leaf disease classification** using deep learning on the **AI-MedLeafX** dataset. In addition to prediction performance, the project also emphasizes **model interpretability** using Explainable AI (XAI) techniques such as **Grad-CAM** and **t-SNE**.

The project compares multiple modern CNN architectures and builds a strong baseline for future agricultural disease detection systems.

### Main Objectives
- Classify diseases on medicinal plant leaves from RGB images.
- Compare multiple CNN architectures on the same dataset.
- Analyze model behavior through XAI.
- Build a strong baseline for future real-world deployment.

---

## ✨ Highlights
- Fine-tuning pretrained CNN models on **AI-MedLeafX**
- Comparison of **ResNet50**, **EfficientNetV2-S**, and **ConvNeXt-Tiny**
- Explainability with **Grad-CAM** and **t-SNE**
- Best reported model accuracy: **98.16%**
- Project notebooks for training, evaluation, and XAI experiments

---

## 🖼️ Sample Images
<p align="center">
  <img src="assets/images/sample_1.png" width="18%" />
  <img src="assets/images/sample_2.png" width="18%" />
  <img src="assets/images/sample_3.png" width="18%" />
  <img src="assets/images/sample_4.png" width="18%" />
  <img src="assets/images/sample_5.png" width="18%" />
</p>

## 🔬 XAI / Result Preview
<p align="center">
  <img src="assets/images/xai_1.png" width="22%" />
  <img src="assets/images/xai_2.png" width="22%" />
  <img src="assets/images/xai_3.png" width="22%" />
  <img src="assets/images/xai_4.png" width="22%" />
</p>

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Highlights](#-highlights)
- [Sample Images](#️-sample-images)
- [XAI / Result Preview](#-xai--result-preview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Models](#-models)
- [Methodology](#️-methodology)
- [Results](#-results)
- [Explainable AI](#-explainable-ai)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Inference on New Images](#-inference-on-new-images)
- [Upload Large Files with Git LFS](#-upload-large-files-with-git-lfs)
- [How to Upload to GitHub](#-how-to-upload-to-github)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [Authors](#-authors)

---

## 🎯 Problem Statement
Early plant disease detection is important for reducing agricultural loss and improving smart farming decisions. Unlike many previous works that focus on common crop plants, this project studies **medicinal plants**, a less explored but important domain.

The main task is to classify leaf images into healthy/disease categories and understand the model’s attention through explainability methods.

---

## 🧪 Dataset
The project uses **AI-MedLeafX**, a medicinal plant disease dataset containing **4 plant species** and **13 classes**.

### Plant Species
- Camphor
- HariTaki
- Neem
- Sojina

### Disease Categories
- Bacterial Spot
- Shot Hole
- Powdery Mildew
- Yellow Leaf
- Healthy Leaf

### Input Configuration
- Image size: **224 × 224**
- RGB images
- ImageNet normalization

### Data Split in Notebook
- Train: 70%
- Validation: 20%
- Test: 10%

### Observed Dataset Size
- **65,178 images**
- **13 classes**
- Train: **45,624**
- Validation: **13,035**
- Test: **6,519**

> Note: The original report mentions the base AI-MedLeafX dataset at around **10,858 images**, while the notebook appears to use an augmented version.

---

## 🧠 Models
### ResNet50
A strong CNN baseline using residual connections.

### EfficientNetV2-S
A more efficient architecture balancing performance and computational cost.

### ConvNeXt-Tiny
A modern ConvNet inspired by Transformer-era design choices, achieving the best result in this project.

---

## ⚙️ Methodology
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
- Resize to `224x224`
- Random horizontal flip
- Random rotation
- Convert to tensor
- Normalize with ImageNet mean/std

### Training Strategy
- Pretrained ImageNet weights
- AdamW optimizer
- Cross-Entropy Loss
- Early stopping
- ReduceLROnPlateau
- Two-stage training strategy

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 📊 Results
| Model | Accuracy (%) | Precision | Recall | F1-score | Params |
|------|-------------:|----------:|-------:|---------:|------:|
| ResNet50 | 95.03 | 0.95 | 0.95 | 0.95 | ~24M |
| EfficientNetV2-S | 96.06 | 0.97 | 0.97 | 0.97 | ~21M |
| ConvNeXt-Tiny | **98.16** | **0.98** | **0.98** | **0.98** | ~28M |

### Observations
- **ConvNeXt-Tiny** achieved the best overall performance.
- **EfficientNetV2-S** balanced performance and model size well.
- **ResNet50** remained a useful and reliable baseline.

---

## 🔍 Explainable AI
### Grad-CAM
Grad-CAM highlights the image regions most influential to the final prediction.

Reported findings:
- Strong localization for **Bacterial Spot** and **Shot Hole**.
- Distributed attention across the leaf for **Healthy Leaf** images.
- More difficulty with **Powdery Mildew** because symptoms are more diffuse.

### t-SNE
`t-SNE` visualizes learned feature embeddings in 2D.

Observations:
- ResNet50 forms moderately separated clusters.
- EfficientNetV2-S provides the clearest cluster separation.
- ConvNeXt-Tiny forms denser clusters but still achieves the best classification accuracy.

---

## 🗂 Project Structure
```text
.
├── README.md
├── README_full_bilingual.md
├── requirements.txt
├── assets/
│   └── images/
├── Nhom_01_NhanDienBenhTrenLaCay.docx
├── medleaf-disease-detection (2).ipynb
├── test.ipynb
├── content/
│   └── ai-medleafx/
├── results/
├── drive_outputs/
├── n/
├── e/
├── EfficientNetV2_S.pth
├── EfficientNetV2_S_final.pth
├── EfficientNetV2_S_stage1_best.pth
└── EfficientNetV2_S_stage2_best.pth
```

---

## 🚀 Installation
```bash
pip install -r requirements.txt
```

### Kaggle API Setup
Place your Kaggle API file at:

```bash
~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## ⚡ Quick Start
```bash
jupyter notebook
```
Open:
- `medleaf-disease-detection (2).ipynb`
- `test.ipynb`

Dataset command used in notebook:
```bash
kaggle datasets download -d mrlocbap/ai-medleafx
```

---

## 🖼 Inference on New Images
The `test.ipynb` notebook includes helper functions for:
- single-image prediction
- Grad-CAM visualization
- LIME explanation
- loading trained `.pth` checkpoints

---

## 📦 Upload Large Files with Git LFS
If you want to upload large files such as `.pth`, `.h5`, `.zip`, or large datasets, use **Git LFS**.

### Install Git LFS
```bash
brew install git-lfs
```

### Enable Git LFS
```bash
git lfs install
```

### Track large file types
```bash
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.zip"
git lfs track "*.pt"
git lfs track "*.ckpt"
```

### Add LFS config and files
```bash
git add .gitattributes
git add EfficientNetV2_S.pth EfficientNetV2_S_final.pth results.zip
```

### Commit and push
```bash
git commit -m "Track large model files with Git LFS"
git push origin main
```

> Important: GitHub has size limits. Git LFS is the correct approach for model checkpoints and large artifacts.

---

## ☁️ How to Upload to GitHub
### Option 1: GitHub Website
1. Create a repository on GitHub.
2. Upload `README.md`, `README_full_bilingual.md`, `requirements.txt`, `assets/`, notebooks, and other selected files.
3. Commit changes.

### Option 2: Git Command Line
```bash
cd "/Users/hongviet/Documents/ComputerVision/đề án cuối kì"
git add README.md README_full_bilingual.md requirements.txt .gitignore assets *.ipynb *.docx
git commit -m "Add bilingual project documentation and notebooks"
git push -u origin main
```

### If you also want to upload large files with Git LFS
```bash
git lfs install
git lfs track "*.pth" "*.h5" "*.zip"
git add .gitattributes
git add *.pth *.h5 *.zip
git commit -m "Add model checkpoints with Git LFS"
git push origin main
```

---

## ⚠️ Limitations
- Training data mostly comes from controlled conditions.
- Diffuse disease patterns remain harder to explain and localize.
- The project has not yet been deployed as a real-time production application.

---

## 🔮 Future Work
- Train on real-world field images
- Try ViT, Swin Transformer, and hybrid architectures
- Extend XAI analysis with SHAP and more detailed evaluation
- Build a user-friendly application
- Deploy on mobile or drone-based platforms

---

## 👥 Authors
| Name | Student ID | Role |
|------|------------|------|
| **Võ Hồng Việt** | `22725461` | Team member |
| **Trần Quang Lộc** | `22732861` | Team member |
| **Trương Lâm Nhựt** | `22721871` | Team member |

---

# Phiên bản tiếng Việt

## 📌 Tổng quan
Dự án tập trung vào bài toán **phân loại bệnh trên lá cây dược liệu** bằng học sâu trên bộ dữ liệu **AI-MedLeafX**. Bên cạnh hiệu năng phân loại, dự án còn chú trọng **khả năng giải thích mô hình** bằng các kỹ thuật Explainable AI (XAI) như **Grad-CAM** và **t-SNE**.

Dự án so sánh nhiều kiến trúc CNN hiện đại và xây dựng một baseline mạnh cho các hệ thống chẩn đoán bệnh cây trong tương lai.

### Mục tiêu chính
- Phân loại bệnh trên lá cây dược liệu từ ảnh RGB.
- So sánh nhiều kiến trúc CNN trên cùng bộ dữ liệu.
- Phân tích cách mô hình ra quyết định bằng XAI.
- Tạo nền tảng cho các ứng dụng thực tế sau này.

---

## ✨ Điểm nổi bật
- Fine-tune các mô hình CNN pretrained trên **AI-MedLeafX**
- So sánh **ResNet50**, **EfficientNetV2-S**, và **ConvNeXt-Tiny**
- Giải thích mô hình bằng **Grad-CAM** và **t-SNE**
- Độ chính xác tốt nhất được báo cáo: **98.16%**
- Có notebook cho huấn luyện, đánh giá và thử nghiệm XAI

---

## 🖼️ Ảnh mẫu
<p align="center">
  <img src="assets/images/sample_1.png" width="18%" />
  <img src="assets/images/sample_2.png" width="18%" />
  <img src="assets/images/sample_3.png" width="18%" />
  <img src="assets/images/sample_4.png" width="18%" />
  <img src="assets/images/sample_5.png" width="18%" />
</p>

## 🔬 Minh họa kết quả / XAI
<p align="center">
  <img src="assets/images/xai_1.png" width="22%" />
  <img src="assets/images/xai_2.png" width="22%" />
  <img src="assets/images/xai_3.png" width="22%" />
  <img src="assets/images/xai_4.png" width="22%" />
</p>

---

## 📚 Mục lục
- [Tổng quan](#-tổng-quan)
- [Điểm nổi bật](#-điểm-nổi-bật)
- [Ảnh mẫu](#️-ảnh-mẫu)
- [Minh họa kết quả / XAI](#-minh-họa-kết-quả--xai)
- [Bài toán](#-bài-toán)
- [Bộ dữ liệu](#-bộ-dữ-liệu)
- [Mô hình](#-mô-hình)
- [Phương pháp](#️-phương-pháp)
- [Kết quả](#-kết-quả)
- [Giải thích mô hình](#-giải-thích-mô-hình)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Chạy nhanh](#-chạy-nhanh)
- [Dự đoán ảnh mới](#-dự-đoán-ảnh-mới)
- [Upload file lớn bằng Git LFS](#-upload-file-lớn-bằng-git-lfs)
- [Cách đưa lên GitHub](#️-cách-đưa-lên-github)
- [Hạn chế](#️-hạn-chế)
- [Hướng phát triển](#-hướng-phát-triển)
- [Tác giả](#-tác-giả)

---

## 🎯 Bài toán
Phát hiện sớm bệnh trên cây giúp giảm thiệt hại nông nghiệp và hỗ trợ quyết định trong nông nghiệp thông minh. Khác với nhiều nghiên cứu trước tập trung vào cây nông nghiệp phổ biến, dự án này nghiên cứu **cây dược liệu** — một hướng còn ít được khai thác.

Mục tiêu chính là phân loại ảnh lá cây thành lá khỏe hoặc các loại bệnh, đồng thời giải thích mô hình đang chú ý vào đâu trên ảnh.

---

## 🧪 Bộ dữ liệu
Dự án sử dụng **AI-MedLeafX**, bộ dữ liệu bệnh lá cây dược liệu gồm **4 loài cây** và **13 lớp**.

### Loài cây
- Camphor
- HariTaki
- Neem
- Sojina

### Nhóm bệnh
- Bacterial Spot
- Shot Hole
- Powdery Mildew
- Yellow Leaf
- Healthy Leaf

### Cấu hình đầu vào
- Kích thước ảnh: **224 × 224**
- Ảnh RGB
- Chuẩn hóa theo ImageNet

### Tỉ lệ chia dữ liệu trong notebook
- Train: 70%
- Validation: 20%
- Test: 10%

### Kích thước quan sát được
- **65,178 ảnh**
- **13 lớp**
- Train: **45,624**
- Validation: **13,035**
- Test: **6,519**

> Lưu ý: báo cáo gốc nêu AI-MedLeafX khoảng **10,858 ảnh**, còn notebook có vẻ đang dùng bản tăng cường dữ liệu.

---

## 🧠 Mô hình
### ResNet50
Mô hình CNN baseline mạnh với residual connection.

### EfficientNetV2-S
Kiến trúc tối ưu hơn về hiệu năng và chi phí tính toán.

### ConvNeXt-Tiny
CNN hiện đại lấy cảm hứng từ các thiết kế thời Transformer và đạt kết quả tốt nhất trong dự án.

---

## ⚙️ Phương pháp
```text
Ảnh đầu vào
   ↓
Tiền xử lý & tăng cường dữ liệu
   ↓
Chia Train / Validation / Test
   ↓
Fine-tune các CNN pretrained
   ↓
Đánh giá bằng các chỉ số
   ↓
Phân tích XAI (Grad-CAM, t-SNE)
```

### Tiền xử lý
- Resize về `224x224`
- Random horizontal flip
- Random rotation
- Chuyển sang tensor
- Chuẩn hóa theo mean/std của ImageNet

### Chiến lược huấn luyện
- Dùng trọng số pretrained từ ImageNet
- AdamW optimizer
- Cross-Entropy Loss
- Early stopping
- ReduceLROnPlateau
- Huấn luyện hai giai đoạn

### Chỉ số đánh giá
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 📊 Kết quả
| Mô hình | Accuracy (%) | Precision | Recall | F1-score | Số tham số |
|------|-------------:|----------:|-------:|---------:|------:|
| ResNet50 | 95.03 | 0.95 | 0.95 | 0.95 | ~24M |
| EfficientNetV2-S | 96.06 | 0.97 | 0.97 | 0.97 | ~21M |
| ConvNeXt-Tiny | **98.16** | **0.98** | **0.98** | **0.98** | ~28M |

### Nhận xét
- **ConvNeXt-Tiny** cho hiệu năng tổng thể cao nhất.
- **EfficientNetV2-S** cân bằng tốt giữa độ chính xác và kích thước mô hình.
- **ResNet50** vẫn là baseline đáng tin cậy.

---

## 🔍 Giải thích mô hình
### Grad-CAM
Grad-CAM giúp làm nổi bật vùng ảnh ảnh hưởng mạnh nhất đến dự đoán của mô hình.

Một số nhận xét từ báo cáo:
- Định vị tốt ở các bệnh như **Bacterial Spot** và **Shot Hole**.
- Với **Healthy Leaf**, vùng chú ý trải đều hơn trên phiến lá.
- Với **Powdery Mildew**, mô hình khó hơn do dấu hiệu bệnh lan tỏa.

### t-SNE
`t-SNE` giúp trực quan hóa không gian đặc trưng mô hình học được.

Quan sát:
- ResNet50 tạo cụm ở mức tương đối.
- EfficientNetV2-S tách cụm rõ nhất.
- ConvNeXt-Tiny tạo cụm dày hơn nhưng vẫn đạt accuracy cao nhất.

---

## 🗂 Cấu trúc thư mục
```text
.
├── README.md
├── README_full_bilingual.md
├── requirements.txt
├── assets/
│   └── images/
├── Nhom_01_NhanDienBenhTrenLaCay.docx
├── medleaf-disease-detection (2).ipynb
├── test.ipynb
├── content/
│   └── ai-medleafx/
├── results/
├── drive_outputs/
├── n/
├── e/
├── EfficientNetV2_S.pth
├── EfficientNetV2_S_final.pth
├── EfficientNetV2_S_stage1_best.pth
└── EfficientNetV2_S_stage2_best.pth
```

---

## 🚀 Cài đặt
```bash
pip install -r requirements.txt
```

### Cấu hình Kaggle API
```bash
~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## ⚡ Chạy nhanh
```bash
jupyter notebook
```
Mở các file:
- `medleaf-disease-detection (2).ipynb`
- `test.ipynb`

Lệnh tải dữ liệu trong notebook:
```bash
kaggle datasets download -d mrlocbap/ai-medleafx
```

---

## 🖼 Dự đoán ảnh mới
Notebook `test.ipynb` có sẵn các hàm hỗ trợ:
- dự đoán ảnh đơn
- trực quan Grad-CAM
- giải thích bằng LIME
- nạp checkpoint `.pth`

---

## 📦 Upload file lớn bằng Git LFS
Nếu bạn muốn đưa các file lớn như `.pth`, `.h5`, `.zip` hoặc artifact nặng lên GitHub, nên dùng **Git LFS**.

### Cài Git LFS
```bash
brew install git-lfs
```

### Kích hoạt Git LFS
```bash
git lfs install
```

### Theo dõi các kiểu file lớn
```bash
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.zip"
git lfs track "*.pt"
git lfs track "*.ckpt"
```

### Add cấu hình LFS và file lớn
```bash
git add .gitattributes
git add EfficientNetV2_S.pth EfficientNetV2_S_final.pth results.zip
```

### Commit và push
```bash
git commit -m "Track large model files with Git LFS"
git push origin main
```

> Lưu ý: GitHub có giới hạn dung lượng file thường, nên Git LFS là cách đúng để upload checkpoint và artifact lớn.

---

## ☁️ Cách đưa lên GitHub
### Cách 1: Dùng giao diện GitHub
1. Tạo repository trên GitHub.
2. Upload các file như `README.md`, `README_full_bilingual.md`, `requirements.txt`, `assets/`, notebook, và các file bạn muốn public.
3. Commit là xong.

### Cách 2: Dùng lệnh Git
```bash
cd "/Users/hongviet/Documents/ComputerVision/đề án cuối kì"
git add README.md README_full_bilingual.md requirements.txt .gitignore assets *.ipynb *.docx
git commit -m "Add bilingual project documentation and notebooks"
git push -u origin main
```

### Nếu muốn upload cả file lớn bằng Git LFS
```bash
git lfs install
git lfs track "*.pth" "*.h5" "*.zip"
git add .gitattributes
git add *.pth *.h5 *.zip
git commit -m "Add model checkpoints with Git LFS"
git push origin main
```

---

## ⚠️ Hạn chế
- Dữ liệu huấn luyện chủ yếu đến từ môi trường kiểm soát.
- Các bệnh có dấu hiệu lan tỏa vẫn khó giải thích và định vị hơn.
- Dự án chưa được triển khai thành ứng dụng thời gian thực.

---

## 🔮 Hướng phát triển
- Huấn luyện trên ảnh thực địa
- Thử ViT, Swin Transformer và mô hình lai
- Mở rộng XAI với SHAP và các đánh giá sâu hơn
- Xây dựng ứng dụng thân thiện với người dùng
- Triển khai trên mobile hoặc drone

---

## 👥 Tác giả
| Họ tên | MSSV | Vai trò |
|------|------------|------|
| **Võ Hồng Việt** | `22725461` | Thành viên nhóm |
| **Trần Quang Lộc** | `22732861` | Thành viên nhóm |
| **Trương Lâm Nhựt** | `22721871` | Thành viên nhóm |
