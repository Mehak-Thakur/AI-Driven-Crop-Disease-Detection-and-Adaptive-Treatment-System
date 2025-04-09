AI-Driven-Crop-Disease-Detection-and-Adaptive-Treatment-System

This project focuses on developing an AI-based model to accurately detect and classify crop diseases using image data. By leveraging transfer learning and deep neural networks, the model aids farmers and agricultural professionals in early identification of plant diseases, leading to timely interventions and improved crop yield.

Project Overview
The goal is to create a robust and efficient machine learning model capable of:

Analyzing plant leaf images

Detecting the presence and type of disease

Assisting in real-time agricultural decision-making

Model Architecture
The model is built using MobileNetV2, a lightweight deep convolutional neural network optimized for mobile and embedded vision applications. The architecture is fine-tuned with:

Data Augmentation

Batch Normalization

Dropout layers to prevent overfitting

Dataset
The dataset contains thousands of labeled images of healthy and diseased leaves across various crop types. The images are preprocessed using:

Resizing and normalization

Image augmentation (rotation, zoom, flip)

Model Training Summary
Initial Training Phase (10 Epochs)
Metric	Result
Best Validation Accuracy	95.15% (Epoch 9)
Best Validation Loss	0.1465
Precision	Up to 96.3%
Recall	Up to 92.5%
These metrics validate the model’s effectiveness in disease classification using visual symptoms.

Evaluation & Results
Confusion Matrix: Visualized to show correct vs incorrect classifications

Accuracy Curve: Steady improvement in both training and validation accuracy

Loss Curve: Gradual decrease confirming proper learning

Features
Transfer Learning with MobileNetV2

High precision and recall for disease detection

Lightweight model suitable for real-time applications

Scalable for multiple crop types

Tools & Technologies
Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib & Seaborn for visualization
