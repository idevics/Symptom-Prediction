# Disease Prediction Using PyTorch

## Overview
This project uses a neural network built with PyTorch to predict possible diseases based on user-provided symptoms.

The model learns from a disease-symptom dataset and returns the top predicted diseases with their estimated probabilities.

## Dataset
The project uses the [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).

## Technologies
- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn

## Features
- Symptom preprocessing and encoding
- Multiclass disease classification
- Neural network training with early stopping
- Model evaluation using accuracy and classification report
- Top-k disease predictions
- Model saving and loading

## Usage

Provide a list of symptoms to get predictions:

```python
user_input = [
    "continuous_sneezing",
    "watering_from_eyes"
]

predictions = predict_disease(user_input, top_k=3)

print(predictions)