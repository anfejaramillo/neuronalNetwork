# TensorFlow Learning Examples 📚🤖

This repository contains two practical examples built with **TensorFlow** and **Keras**.  
Both scripts demonstrate fundamental deep learning concepts:

1. **Image classification using Fashion MNIST**
2. **Linear regression for temperature conversion (Celsius to Fahrenheit)**

These examples are ideal for understanding the basics of neural networks, training processes, loss visualization, and model persistence.

---

# 📂 Repository Structure

.
├── fashion_mnist_classifier.py
├── celsius_to_fahrenheit_model.py
└── README.md


---

# 1️⃣ Fashion MNIST Image Classification

## 📌 Overview

This script trains a neural network to classify clothing images from the **Fashion MNIST** dataset provided by TensorFlow Datasets.

Dataset:  
- 28x28 grayscale images  
- 10 clothing categories  
- 60,000 training images  
- 10,000 testing images  

---

## 🧠 Model Architecture

The model is built using `tf.keras.Sequential` with the following layers:

- **Flatten Layer** → Converts 28x28 images into a 1D vector
- **Dense (50 neurons, ReLU)**
- **Dense (50 neurons, ReLU)**
- **Dense (10 neurons, Softmax)** → Output probabilities for 10 classes

---

## ⚙️ Training Configuration

- Optimizer: `Adam`
- Loss Function: `SparseCategoricalCrossentropy`
- Metric: `Accuracy`
- Epochs: `5`
- Batch Size: `32`

---

## 🔄 Data Processing

- Images are normalized to values between `0` and `1`
- Dataset is cached for performance
- Data is shuffled and batched before training

---

## 📊 Visualization

After training, the script:

- Makes predictions on test images
- Displays:
  - The image
  - Predicted label
  - Confidence percentage
  - Probability distribution graph

Correct predictions appear in **blue**  
Incorrect predictions appear in **red**

---

## 💾 Model Persistence

The trained model is saved as:

modelTopologyAndWeights.h5


This file contains:
- Model architecture
- Trained weights
- Optimizer state

You can reload it using:

```python
tf.keras.models.load_model("modelTopologyAndWeights.h5")
2️⃣ Celsius to Fahrenheit Linear Regression
📌 Overview
This script trains a neural network to learn the mathematical relationship between:

Fahrenheit = (Celsius × 9/5) + 32
Instead of explicitly coding the formula, the model learns it from example data.

🧠 Model Architecture
One Dense layer

One neuron

Input shape: [1]

This represents a simple linear regression model.

⚙️ Training Configuration
Optimizer: Adam (learning rate = 0.1)

Loss Function: Mean Squared Error

Epochs: 1000

Silent training (verbose=False)

📊 Loss Visualization
After training, the script plots:

Loss vs Epoch

This allows you to see how the model improves over time.

🔍 Prediction Example
The model predicts the Fahrenheit value for:

100°C
Expected output should be close to:

212°F
🚀 How to Run
1️⃣ Install Dependencies
pip install tensorflow tensorflow-datasets numpy matplotlib
2️⃣ Run Fashion MNIST Classifier
python fashion_mnist_classifier.py
3️⃣ Run Celsius to Fahrenheit Model
python celsius_to_fahrenheit_model.py
📖 Concepts Covered
Neural Networks

Dense Layers

Activation Functions (ReLU, Softmax)

Loss Functions

Model Training

Dataset Normalization

Visualization with Matplotlib

Model Saving and Loading

Linear Regression with Neural Networks

🎯 Purpose of This Repository
This repository is designed for:

Beginners learning TensorFlow

Students exploring neural networks

Developers reviewing core ML concepts

Educational demonstrations

It provides clear and minimal examples of both:

Classification problems

Regression problems

📌 Requirements
Python 3.8+

TensorFlow 2.x

TensorFlow Datasets

NumPy

Matplotlib

🏁 Final Notes
These scripts are intentionally simple and educational.
They focus on clarity rather than production optimization.

Feel free to extend them by:

Increasing epochs

Adding more layers

Implementing callbacks

Exporting models in different formats

Evaluating performance metrics in more detail

Happy Learning! 🚀