---
title: OralCancerDemo
emoji: 📊
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
---

# 🦷 Oral Health Condition Classifier

A deep learning project for **oral disease classification** using **Vision Transformer (ViT)**, with earlier experiments on **MobileNet** and **DenseNet121**.  
The final model was trained on a custom dataset and deployed as a demo on **Hugging Face Spaces** with a Gradio interface.

---

## 🚀 Features
- Classifies 6 oral health conditions:
  - Calculus  
  - Data caries  
  - Gingivitis  
  - Mouth Ulcer  
  - Tooth Discoloration  
  - Hypodontia  
- Built with **PyTorch** + **timm** (ViT models).  
- Interactive **Gradio demo** for real-time predictions.  
- Supports **Grad-CAM visualization** for model explainability.  
- Includes evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

---

## 📂 Project Structure


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
