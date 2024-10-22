# Cyber Security: URL Phishing Detection

## Project Overview
This project aims to detect phishing URLs to safeguard users from fraudulent websites that could compromise sensitive information. Specifically, it targets the detection of malicious URLs in advertisements displayed on the Book-My-Show platform, ensuring that users are protected from phishing attacks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [How to Use](#how-to-use)
- [Installation](#installation)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
Phishing is a major cyber threat that tricks users into revealing confidential information, such as login credentials and financial data. This project addresses this challenge by developing a machine learning model to classify URLs into:
- **Phishing (Malicious)**
- **Suspicious**
- **Legitimate**

Our goal is to enhance the security of the Book-My-Show platform by identifying and filtering malicious links from its advertisements.

## Dataset
The dataset used in this project contains approximately 11,000 URLs with 32 features that help in classifying URLs into the following categories:
- **1**: Phishing (Malicious)
- **0**: Suspicious
- **-1**: Legitimate

## Technologies Used
- **Python**: For data processing and machine learning
- **Pandas**: Data manipulation and exploration
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Building machine learning models
- **XGBoost**: Gradient boosting algorithm
- **Keras**: Building artificial neural networks
- **SciKeras**: Scikit-learn wrapper for Keras models

## Project Workflow
1. **Data Loading & Exploration**: Loading the dataset and performing basic data exploration, checking for null values, and understanding the distribution of features.
2. **Data Visualization**: Visualizing feature distributions and correlations using heatmaps to identify patterns.
3. **Feature Engineering**: Removing highly correlated features to reduce redundancy and improve model efficiency.
4. **Model Building & Evaluation**: Training multiple machine learning models and evaluating their performance using accuracy, precision, recall, and F1-score.
5. **Hyperparameter Tuning**: Using `RandomizedSearchCV` and `GridSearchCV` to fine-tune the hyperparameters for the best-performing model (XGBoost).
6. **Model Evaluation**: Generating classification reports, confusion matrices, and ROC curves for detailed model performance assessment.

## Model Performance
After testing multiple models, the **XGBoost Classifier** performed the best, achieving:
- **Accuracy**: 97%
- **Precision (Phishing)**: 98%
- **Recall (Phishing)**: 95%

The detailed performance of other models such as Random Forest, Bagging Classifier, and Neural Networks is also evaluated and provided in the notebook.

## How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/megokul/Cyber-Security-URL-Phishing-Detection.git

## Installation
Ensure you have the following dependencies installed before running the notebook:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras tensorflow scikeras mlxtend
```
Alternatively, you can install all required dependencies at once using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Future Improvements
- **Real-time Phishing Detection**: Incorporate real-time detection to classify URLs as they are entered.
- **Ensemble Methods**: Implement advanced ensemble techniques for further accuracy improvements.
- **Explainability**: Use tools like SHAP for better model interpretability.
- **Deep Learning Models**: Experiment with deep learning techniques, such as RNNs or LSTMs, to capture URL patterns more effectively.

## License
This project is licensed under the MIT License.

