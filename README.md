# Detection of Deepfake Media for Secure Communication 

An automated deepfake detection system that uses a Deep Neural Network (DNN) to classify audio-visual media as **"Real"** or **"Fake"**, with a Flask-based web interface for real-time analysis.

## Overview

Deepfake technology, powered by deep learning models like GANs and autoencoders, has made it increasingly easy to generate realistic synthetic video and audio that convincingly mimic real individuals. This poses serious risks to digital trust and security across banking, law enforcement, media, and everyday communication.

This project implements a real-time deepfake detection pipeline built on a DNN trained on the **FakeAVCeleb** dataset. It detects faces frame-by-frame using OpenCV and Haar cascades, extracts and preprocesses features, and classifies each sample as real or fake with a confidence score — deployed through a lightweight Flask web application.

## Features

- Frame-by-frame face detection using OpenCV and Haar cascade classifiers
- Deep Neural Network classifier with dense layers, ReLU activations, dropout, and batch normalization
- Binary classification (Real / Fake) via a SoftMax output layer
- Real-time inference through a Flask web app (video upload or webcam input)
- Confidence score displayed alongside each prediction
- Evaluation via accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC

## System Architecture

**Pipeline:** `MP4 Video → Frame Extraction → Audio Separation → Feature Engineering → Model Inference → Real/Fake Prediction → Confidence Score`

**DNN Architecture:**
- Input layer — preprocessed feature vectors (58-D)
- Dense Layer 1 — 256 neurons, ReLU
- Batch Normalization
- Dropout — 50%
- Dense Layer 2 — 128 neurons, ReLU
- Dense Layer 3 — 64 neurons, ReLU
- Output Layer — SoftMax activation (binary classification)

**Deployment pipeline:** `Training → Model Export → Flask API → Web Interface → User Feedback`

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, Haar Cascade classifiers |
| Web Framework | Flask |
| Dataset | FakeAVCeleb |
| Supporting Libraries | NumPy, Pandas, Matplotlib, Scikit-learn |

## Dataset

The model is trained and evaluated on the **FakeAVCeleb** dataset, which contains both authentic and AI-generated (synthetic) audiovisual samples, split in an 80:20 train/test ratio.

## Model Training

- Optimizer: Adam
- Loss function: Categorical cross-entropy
- Batch size: 32
- Epochs: up to 50 (with early stopping and learning-rate reduction)
- Regularization: Dropout (0.5) and batch normalization to reduce overfitting

## Results

- Test accuracy: **>90%**
- ROC-AUC: **~0.92**
- High F1-score with balanced precision and recall
- Confusion matrix shows a low rate of false positives and false negatives
- Training/validation curves show steady convergence with minimal overfitting

## Web Application

The Flask web app allows users to:
1. Upload an MP4 video or activate a webcam feed
2. Automatically extract and process frames
3. View the prediction ("Real" or "Fake") with an associated confidence score

## Installation

```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### Requirements (suggested)

```
tensorflow
keras
opencv-python
flask
numpy
pandas
matplotlib
scikit-learn
```

## Usage

```bash
python app.py
```

Then open `http://localhost:5000` in your browser, upload a video or enable your webcam, and click **Analyze** to get the Real/Fake prediction with a confidence score.

## Project Structure

```
├── app.py                 # Flask application entry point
├── model/                 # Trained DNN model files
├── static/                # CSS, JS, and frontend assets
├── templates/             # HTML templates
├── preprocessing/         # Frame extraction, face detection, feature engineering
├── haarcascades/          # Haar cascade XML files
├── requirements.txt
└── README.md
```

## Future Scope

- Support for mobile-compressed and low-resolution video formats
- Multimodal fusion of audio and visual features for improved accuracy
- Optimization for deployment on mobile and edge devices
- More responsive, user-friendly frontend using HTML, CSS, and JavaScript
- Continuous learning to adapt to evolving deepfake generation techniques

## Applications

- Digital forensics and law enforcement
- Secure banking and financial transaction verification
- Media and social platform content moderation
- Public awareness and media verification tools

## Authors

- Harishma R — Department of Electronics and Telecommunication Engineering, R V College of Engineering, Bengaluru
- K Sreelakshmi — Department of Electronics and Telecommunication Engineering, R V College of Engineering, Bengaluru

## Citation

If you use this work, please cite:

> Harishma R and K Sreelakshmi, "Detection of Deepfake Media for Secure Communication Using Deep Learning," 2025 9th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS), DOI: 10.1109/CSITSS67709.2025.11295707.

## License

Specify your project's license here (e.g., MIT, Apache 2.0).
