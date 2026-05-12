# Gender Recognition Using Voice Analysis

## Overview
This project is a real-time voice-based gender recognition system developed using a hybrid deep learning approach. The system analyzes voice recordings and predicts gender as male or female using acoustic voice features and a trained Multilayer Perceptron (MLP) model built with PyTorch.

The application allows users to upload WAV audio files or record live audio directly through the browser. Extracted voice features are processed by the trained model to generate predictions along with confidence scores.

To improve prediction reliability, the system combines deep learning predictions with pitch-based calibration logic, helping reduce overconfident outputs and improving borderline case handling.

---

## Features
- Real-time gender prediction
- Upload WAV audio files
- Live microphone recording
- Acoustic feature extraction
- Confidence score generation
- Hybrid deep learning approach
- Interactive web interface
- Fast and lightweight prediction system

---

## Technologies Used
- Python
- PyTorch
- FastAPI
- NumPy
- SQLite
- HTML
- CSS
- JavaScript
- Web Audio API

---

## Model Details
The model was trained using:
- 3168 voice samples
- 20 acoustic voice features

### Extracted Features
- Pitch
- RMS Energy
- Zero Crossing Rate

### Deep Learning Model
- Multilayer Perceptron (MLP)
- Implemented using PyTorch
- Binary classification (Male/Female)

### Optimization Techniques
- Binary Cross Entropy Loss
- Adam Optimizer
- Feature Normalization
- Probability Calibration using Pitch Logic

---

## System Workflow
1. User uploads or records audio
2. Audio is processed and cleaned
3. Acoustic features are extracted
4. Features are normalized
5. Features are passed into the trained MLP model
6. Model predicts gender probabilities
7. Pitch-based calibration improves reliability
8. Final prediction and confidence scores are displayed

---

## Project Structure

```text
gender-recognition-using-voice-analysis/
│
├── backend/
├── frontend/
├── models/
├── dataset/
├── app.py
├── requirements.txt
├── README.md
└── static/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/gender-recognition-using-voice-analysis.git
```

### Move into Project Directory

```bash
cd gender-recognition-using-voice-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python app.py
```

The application will start locally and can be accessed through the browser.

---

## Output
The system displays:
- Predicted Gender
- Confidence Score
- Real-time prediction results

Example:
```text
Predicted Gender: Female
Confidence Score: 92%
```

---

## Applications
- Voice-based AI systems
- Smart assistants
- Speech analytics
- Human-computer interaction
- Educational AI projects
- Audio classification systems

---

## Future Improvements
- CNN and LSTM based architectures
- Better noisy audio handling
- Larger and more diverse datasets
- Spectrogram visualization
- Multi-class voice classification
- Mobile application support
- Transformer-based audio models

---

## Conclusion
The project successfully develops a voice-based gender recognition system using a hybrid deep learning approach. A trained MLP model is used to classify gender based on acoustic features, achieving strong performance.

A calibration mechanism based on pitch is applied to improve real-time prediction reliability and handle borderline cases. The system provides accurate and fast results through a user-friendly interface.

Overall, the project demonstrates an effective combination of deep learning and domain knowledge for practical voice analysis applications.

Centre for Artificial Intelligence  
Madhav Institute of Technology & Science, Gwalior
