# Machine Learning with MNIST

This project implements basic machine learning models and training procedures, primarily for the MNIST dataset and a simple sine regression task. The focus is on understanding and building foundational regression and classification models from scratch using Python and NumPy.

## Getting Started

1. Place the MNIST PNG dataset in the `mnist_png/` directory or update the path in `util.py`.
2. Install dependencies:
   ```
   pip install numpy matplotlib pillow
   ```
3. Run the main script:
   ```
   python model.py
   ```
   This will run the regression and classification routines, producing plots and metrics.

## Overview

- **Custom Dataset Utilities:**  
  Synthetic datasets for regression and logistic tasks, as well as wrappers for MNIST image data.

- **Training and Evaluation Loops:**  
  Each model supports training with stochastic gradient descent, and evaluation using loss and/or accuracy metrics. Includes plotting utilities for loss/accuracy curves and confusion matrices.

### PolynomialRegressionModel

Linear regression with polynomial features, supporting customizable degree and learning rate.

- Fits a polynomial model:  
  \( y = w_0 + w_1 x + w_2 x^2 + \ldots + w_d x^d \)
- Implements mean squared error loss and gradient descent.


- `linear_regression()` Runs polynomial regression on synthetic sine data. Compares different learning rates.

### BinaryLogisticRegressionModel

  Logistic regression for binary classification (e.g., distinguishing between two MNIST digits).

- Fits a logistic regression model for binary targets.
- Uses image pixels as features (flattened for MNIST).
- Implements cross-entropy loss and stochastic gradient descent.


- `binary_classification()` Trains and evaluates binary logistic regression on MNIST, including accuracy plots and confusion matrix.

### MultiLogisticRegressionModel

  Softmax regression for multi-class classification (e.g., digit recognition across all MNIST classes 0-9).

- Fits a multinomial (softmax) logistic regression for multi-class targets.
- Each class has its own weight vector.
- Uses cross-entropy loss and supports prediction across multiple classes.


- `multi_classification()` Trains and evaluates multinomial logistic regression on MNIST, with support for accuracy, confusion matrix, and visualizing learned weights.




