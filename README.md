# deep_learning_siamese_facial_recognition

This repository contains the implementation of Assignment 2 in the Deep Learning course at Ben-Gurion University.  
The project implements a Siamese Convolutional Neural Network (CNN) for one-shot facial recognition using the LFW-a dataset.

The assignment focuses on:
- Implementing a Siamese neural network for one-shot learning  
- Using convolutional layers to extract hierarchical features  
- Experimenting with optimizers (Adam, SGD), batch normalization, dropout, and L2 regularization  
- Analyzing performance with logging tools and reporting results  

---

## Project Structure

- `siamese_face_recognition.ipynb` – Jupyter Notebook with the implementation and experiments  
- `siamese_face_recognition_report.pdf` – Report summarizing design, implementation details, and results  

---

## Dataset

- Labeled Faces in the Wild (LFW-a) dataset was used.  
- Train/Test split provided by `pairsDevTrain.txt` and `pairsDevTest.txt`.  
- Images resized to 105x105 pixels and transformed into tensors.  
- Data split into training (80%), validation (20%), and test sets.  

---

## Model Implementation

- Convolutional layers: 64 → 128 → 256 filters, each with ReLU activation and MaxPooling  
- Flattening layer to connect CNN outputs to fully connected layers  
- Fully connected layers: 4096-dim feature vectors  
- Similarity score: computed as absolute difference → Sigmoid activation → Probability (same/different)  
- Loss function: Binary Cross Entropy Loss (BCELoss)  
- Optimizers: Adam and SGD tested (Adam performed better)  
- Batch normalization: improved validation/test accuracy  
- Dropout: tested (0.1, 0.2) but did not improve results  
- L2 regularization: improved generalization, best performance with batch size=32 and weight decay=1e-5  

---

## Training Settings

- Learning rate: 0.0001 with StepLR scheduler (decay by 0.1 every 10 epochs)  
- Batch sizes: 32, 64, 128, 256 tested  
- Epochs: up to 20 with early stopping (patience=5)  
- Best configuration: Batch size = 32, weight decay = 1e-5  

---

## Results

- Best validation accuracy: 69.7%  
- Best test accuracy: 70.4%  
- Batch normalization significantly improved performance  
- Adam outperformed SGD in both accuracy and convergence speed  
- Dropout had no positive effect in this setup  

---

## Setup

Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## Run
1. Open the Jupyter notebook: siamese_face_recognition.ipynb
2. Run all cells to preprocess the dataset, train the Siamese CNN, and evaluate results.
3. Logs and graphs will be generated to show training/validation loss and accuracy.
4. Final test accuracy and example predictions are reported.
