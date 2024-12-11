# HW4-AutoML

This project demonstrates how to optimize machine learning models using **PyCaret**, **Optuna**, and **TPOT AutoML**. It covers hyperparameter tuning, feature engineering, model selection, and ensemble learning on the Titanic dataset.

## [GPT History](https://chatgpt.com/share/675917ea-a1b0-800c-81b5-2620a592d226)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Workflow](#code-workflow)
6. [Results](#results)

---

## Introduction

Machine learning model optimization is critical for achieving high-performing models. This project showcases multiple optimization strategies using popular tools like:

- **PyCaret**: Simplifies ML workflows, enabling comparison, tuning, and ensembling.
- **Optuna**: A hyperparameter optimization framework with flexible objectives.
- **TPOT AutoML**: Automates the ML pipeline using genetic programming.

We demonstrate these tools on the Titanic dataset to classify survivors based on passenger information.

---

## Features

- **End-to-end ML workflow**: Data preprocessing, model selection, hyperparameter tuning, and evaluation.
- **Multi-tool approach**: Compare different AutoML libraries.
- **Flexible optimization**: Optuna and TPOT provide fine-grained control over optimization.
- **Ensemble learning**: Blend and stack models for improved performance.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Required Libraries

Run the following command to install dependencies:

```bash
pip install pycaret[full] optuna tpot pandas scikit-learn
```

---

## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/IdONTKnowCHEK/HW4-AutoML.git
   cd HW4-AutoML
   ```

2. **Run the Jupyter Notebook**

   Launch Jupyter Notebook and open the file:

   ```bash
   jupyter notebook PyCaret_Ensemble_Optimization.ipynb
   ```

3. **Follow the Notebook Steps**

   Execute the cells in the notebook to:
   - Load and preprocess the Titanic dataset.
   - Compare ML algorithms with PyCaret.
   - Tune hyperparameters with PyCaret or Optuna.
   - Optimize pipelines using TPOT AutoML.
   - Evaluate the best models.

---

## Code Workflow

### 1. Data Preprocessing

Prepare the Titanic dataset by:
- Dropping irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- Filling missing values.
- Encoding categorical variables.

### 2. PyCaret Model Comparison and Tuning

- **Setup**: Initialize PyCaret for classification.
- **Compare Models**: Identify the best-performing algorithms.
- **Tune Hyperparameters**: Use Optuna via PyCaret to optimize the top models.

### 3. Optuna Custom Optimization

- Define an objective function to optimize hyperparameters for a `RandomForestClassifier`.
- Use **Optuna** to search for the best hyperparameters.

### 4. AutoML with TPOT

- Use TPOT to automate model selection and hyperparameter tuning.
- Export the best pipeline.

### 5. Ensemble Learning

- Blend and stack top-performing models using PyCaret.

---

## Results

- **Baseline Accuracy**: Achieved using PyCaret's default settings.
- **Optimized Accuracy**: Improved accuracy using hyperparameter tuning and ensemble learning.
- **Pipeline Export**: Exported pipelines for production deployment.

---

## Example Code Snippets

### PyCaret Model Tuning

```python
from pycaret.classification import setup, compare_models, tune_model, finalize_model

# Initialize PyCaret
clf_setup = setup(data=titanic_data, target='Survived', session_id=42, silent=True)

# Compare Models
best_model = compare_models()

# Tune the Best Model
tuned_model = tune_model(best_model, optimize='Accuracy')
```

### Optuna Hyperparameter Optimization

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```
