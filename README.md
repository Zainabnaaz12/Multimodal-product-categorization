# Multimodal Product Categorization

## Overview
This project implements a multimodal deep learning system that classifies e-commerce products by combining information from product images and text descriptions.  
By leveraging computer vision (CNNs) for image features and transformer-based NLP models for text embeddings, the model provides robust product categorization even when one modality (image or text) is noisy or incomplete.

## Problem Statement
Accurate product categorization is essential for search, recommendations, and inventory management in e-commerce platforms. Traditional models rely on either text or image data, which can lead to misclassification.  
This project addresses the problem by learning from both text and image data simultaneously.

## Key Features
- **Image Pipeline:**  
  - Preprocessed images using OpenCV  
  - Extracted image features using ResNet (CNN) pretrained on ImageNet
- **Text Pipeline:**  
  - Tokenized product titles and descriptions  
  - Generated embeddings using BERT from Hugging Face
- **Fusion Layer:**  
  - Combined image and text features to build a multimodal classifier  
  - Implemented in PyTorch
- **Evaluation:**  
  - Train/validation/test split  
  - Evaluated with accuracy and F1-score
- **Scalable Architecture:**  
  - Modular design for adding new modalities or models

## Tech Stack
- Python  
- PyTorch  
- Hugging Face Transformers  
- OpenCV, NumPy, Pandas  
- Scikit-learn

## Results (Work in Progress)
- Current prototype shows improvement in classification accuracy compared to single-modality models.
- More experiments and detailed results will be added as the project progresses.

## Future Work
- Experiment with different transformer models (RoBERTa, DistilBERT)
- Fine-tune CNN on domain-specific product images
- Deploy model using Flask or FastAPI for inference

## How to Run
1. Clone this repository
2. Install requirements:
   pip install -r requirements.txt
3. Add dataset (images + text CSV) to the data/ folder.
4. Run training:
   python train.py
