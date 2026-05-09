import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    })

add_md("""
# 🎓 Machine Learning Project: Resume Text Classification

### Part 1: Introduction
**Student Details:**
* First Name: [Your First Name]
* Last Name Initial: [Your Last Name Initial]
* Last 4 Digits of ID: [1234]

**AI Chatbot Prompts Used:**
* *Prompt 1:* "How to apply POS-aware lemmatization using NLTK in pandas..."
* *Prompt 2:* "How to extract feature importances from a multiclass Logistic Regression..."
* *Goal of usage:* To assist with syntax optimization and generating explainability visualizations.

**Problem & Dataset Explanation:**
This project focuses on **Text Classification** (Supervised Learning). The dataset used is the Resume Classification dataset from Kaggle, which contains thousands of raw resume text strings alongside their corresponding job category (e.g., Finance, IT, Healthcare). The goal is to build an NLP pipeline that cleans the raw text, extracts meaningful numerical features (using TF-IDF), and trains a machine learning model to accurately predict the job category of unseen resumes based purely on their textual content.
""")

add_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

sns.set_theme(style="whitegrid")
""")

add_md("""
### Loading the Dataset and Splitting (Train/Test)
To absolutely prevent any data leakage, we will load the dataset and immediately split it into a `train_set` and `test_set` before applying any feature engineering or vectorization.
""")

add_code("""
# Load dataset
df = pd.read_csv('Resume.csv')

# Basic cleaning of empty rows
if 'Resume_html' in df.columns:
    df = df.drop(columns=['Resume_html'])
df = df.dropna(subset=['Resume_str', 'Category'])
df = df.drop_duplicates(subset=['Resume_str']).reset_index(drop=True)
df['Category'] = df['Category'].str.strip().str.upper()

# Train-Test Split (80/20)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])

print(f"Train set shape: {df_train.shape}")
print(f"Test set shape: {df_test.shape}")

print("\\nFirst 5 rows of Train Set:")
display(df_train.head(5))

print("\\nFirst 5 rows of Test Set:")
display(df_test.head(5))
""")

add_md("""
### Part 6D (Bonus): Special Adjustments for Imbalanced Data
The original dataset has 24 highly imbalanced and highly specific categories (e.g., 'Aviation' and 'Agriculture' have very few samples compared to 'Information-Technology'). 
To treat this imbalance robustly, we apply **Category Grouping** (Dimensionality Reduction of the Target Variable). We map the 24 classes into 9 broader, semantically balanced macro-categories.
""")

add_code("""
print("Class sizes BEFORE mapping (24 Highly Imbalanced Categories):")
print(df_train['Category'].value_counts())
print("\\n" + "-"*50 + "\\n")

category_mapping = {
    'INFORMATION-TECHNOLOGY': 'Technology',
    'BUSINESS-DEVELOPMENT': 'Business & Operations',
    'SALES': 'Business & Operations',
    'BPO': 'Business & Operations',
    'CONSULTANT': 'Business & Operations',
    'FINANCE': 'Finance & Accounting',
    'ACCOUNTANT': 'Finance & Accounting',
    'BANKING': 'Finance & Accounting',
    'ENGINEERING': 'Engineering & Infrastructure',
    'CONSTRUCTION': 'Engineering & Infrastructure',
    'AUTOMOBILE': 'Engineering & Infrastructure',
    'AVIATION': 'Engineering & Infrastructure',
    'DESIGNER': 'Creative & Media',
    'DIGITAL-MEDIA': 'Creative & Media',
    'PUBLIC-RELATIONS': 'Creative & Media',
    'ARTS': 'Creative & Media',
    'APPAREL': 'Creative & Media',
    'HEALTHCARE': 'Healthcare & Wellness',
    'FITNESS': 'Healthcare & Wellness',
    'TEACHER': 'Education',
    'ADVOCATE': 'Legal & HR',
    'HR': 'Legal & HR',
    'CHEF': 'Food & Agriculture',
    'AGRICULTURE': 'Food & Agriculture'
}

# Apply mapping to train and test
df_train['Macro_Category'] = df_train['Category'].map(category_mapping)
df_test['Macro_Category'] = df_test['Category'].map(category_mapping)

df_train = df_train.dropna(subset=['Macro_Category']).reset_index(drop=True)
df_test = df_test.dropna(subset=['Macro_Category']).reset_index(drop=True)

y_train = df_train['Macro_Category']
y_test = df_test['Macro_Category']

print("Class sizes AFTER mapping (9 Balanced Macro Categories):")
print(y_train.value_counts())
""")

add_md("""
### Part 2: Feature Engineering
We will apply text normalization, POS-Aware Lemmatization, and TF-IDF vectorization.
We will explicitly show 3 examples from the train set going through the pipeline.
""")

add_code("""
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_text(text):
    text = str(text).lower()
    text = re.sub('http\\S+\\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\\S+', '', text)
    text = re.sub('@\\S+', '  ', text)
    text = re.sub('[%s]' % re.escape(\"\"\"!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\"\"\"), ' ', text)
    text = re.sub(r'[^\\x00-\\x7f]', r' ', text) 
    text = re.sub('\\s+', ' ', text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    pos_tokens = pos_tag(tokens)
    cleaned = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_tokens]
    return ' '.join(cleaned)

print("Applying Feature Engineering (Cleaning)...")
df_train['Cleaned'] = df_train['Resume_str'].apply(clean_text)
df_test['Cleaned'] = df_test['Resume_str'].apply(clean_text)

print("\\nShowing 3 examples of Feature Engineering (Original vs Cleaned):")
for i in range(3):
    print(f"\\n--- Example {i+1} ---")
    print(f"ORIGINAL (first 100 chars): {df_train['Resume_str'].iloc[i][:100]}...")
    print(f"CLEANED  (first 100 chars): {df_train['Cleaned'].iloc[i][:100]}...")
""")

add_code("""
# TF-IDF Vectorization
# We fit ONLY on the train set to absolutely prevent data leakage
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train = tfidf.fit_transform(df_train['Cleaned'])
X_test = tfidf.transform(df_test['Cleaned'])

print(f"\\nTF-IDF Train Matrix Shape: {X_train.shape}")
print(f"TF-IDF Test Matrix Shape: {X_test.shape}")
""")

add_md("""
### Part 3: Implementation of Learning Algorithms
Instead of using scikit-learn's built-in model, we implement a **Custom Logistic Regression** from scratch. 
Because this is a multi-class problem (9 classes), we implement **Multinomial Logistic Regression (Softmax Regression)**. We use Gradient Descent to optimize the weights, and we ensure the implementation is compatible with `scipy.sparse` matrices (outputted by TF-IDF) to prevent memory crashes and execute quickly.
""")

add_code("""
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, max_iter=1000, C=1.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C # Inverse of regularization strength
        self.classes_ = None
        self.W = None
        self.b = None

    def fit(self, X, y):
        # Identify unique classes
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        num_samples, num_features = X.shape
        
        # One-hot encode the target labels
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])
        Y_one_hot = np.zeros((num_samples, num_classes))
        Y_one_hot[np.arange(num_samples), y_idx] = 1
        
        # Initialize weights and bias
        self.W = np.zeros((num_features, num_classes))
        self.b = np.zeros(num_classes)
        
        # Gradient Descent Optimization
        for i in range(self.max_iter):
            # Calculate linear predictions (compatible with sparse X)
            linear_model = X.dot(self.W) + self.b
            
            # Apply Softmax for multi-class probabilities
            # Shift linear model for numerical stability
            shifted_linear = linear_model - np.max(linear_model, axis=1, keepdims=True)
            exp_scores = np.exp(shifted_linear)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Calculate error
            error = probs - Y_one_hot
            
            # Calculate gradients with L2 Regularization
            # X.T is sparse. sparse.dot(dense) returns dense array.
            dW = (X.T.dot(error) / num_samples) + (self.W / self.C) / num_samples
            db = np.sum(error, axis=0) / num_samples
            
            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
        return self

    def predict_proba(self, X):
        linear_model = X.dot(self.W) + self.b
        shifted_linear = linear_model - np.max(linear_model, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_linear)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        y_idx = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in y_idx])

    @property
    def coef_(self):
        # Transpose W to match scikit-learn's coef_ shape (n_classes, n_features)
        return self.W.T
""")

add_md("""
### Part 6A, 6C, 6G (Bonuses): Hyperparameter GridSearch, K-Fold CV, and Results Presentation
We use `GridSearchCV` with 5-Fold Stratified Cross-Validation (`cv=5`). We will test different hyperparameters for our `CustomLogisticRegression`, specifically the `learning_rate` and `C` (regularization strength).
""")

add_code("""
# Define hyperparameter grid for Custom Logistic Regression
param_grid = {
    'learning_rate': [5.0, 10.0, 20.0],
    'C': [1.0, 10.0]
}

# K-Fold CV GridSearch
grid_search = GridSearchCV(
    CustomLogisticRegression(max_iter=3000),
    param_grid,
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1,
    return_train_score=False
)

print("Running 5-Fold Cross Validation GridSearch...")
grid_search.fit(X_train, y_train)

# Part 6G: Show all permutations in a DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['param_learning_rate', 'param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_df = results_df.rename(columns={'mean_test_score': 'Mean_Macro_F1'})
results_df = results_df.sort_values(by='rank_test_score')

print("\\nAll Hyperparameter Permutations and their K-Fold Macro F1 Scores:")
display(results_df)

print(f"\\nBest Permutation: {grid_search.best_params_}")
print(f"Best Macro F1 Score (Validation): {grid_search.best_score_:.4f}")
""")

add_md("""
### Part 4: Training with the Best Combination
We now extract the best model from the GridSearch (which is already retrained on the entire `trainset` by default via `refit=True`) to be ready for testing.
""")

add_code("""
best_model = grid_search.best_estimator_
print(f"Model selected for final testing: {best_model}")
""")

add_md("""
### Part 6F (Bonus): Explainability (Feature Importance)
To understand what our Logistic Regression model actually learned, we extract the feature coefficients. By matching the coefficients to the TF-IDF vocabulary, we can reveal the **Top 5 most important words** for predicting specific classes.
""")

add_code("""
# Extract feature names from TF-IDF
feature_names = np.array(tfidf.get_feature_names_out())

# Extract coefficients from the best Logistic Regression model
coefs = best_model.coef_

# Let's look at the top 5 words for a few categories
classes_to_explain = ['Technology', 'Finance & Accounting', 'Healthcare & Wellness']

print("Explainability: Top 5 strongest predicting words per category\\n")
for cls in classes_to_explain:
    class_index = list(best_model.classes_).index(cls)
    top5_indices = coefs[class_index].argsort()[-5:][::-1]
    top5_words = feature_names[top5_indices]
    print(f"Top 5 words for {cls}:")
    print(f"-> {', '.join(top5_words)}\\n")
""")

add_md("""
### Part 5: Prediction and Evaluation on Test Set
Finally, we predict on the `test_set` and evaluate the model using the **Macro-Average F1-Score** as mandated by the quality metric guidelines.
We will explicitly show the prediction for the first 5 test examples.
""")

add_code("""
# Predict on the first 5 examples
sample_test = X_test[:5]
sample_true = y_test.iloc[:5].values
sample_pred = best_model.predict(sample_test)

print("Predictions on the first 5 Test examples:")
for i in range(5):
    print(f"Example {i+1}: True = {sample_true[i]:<30} | Predicted = {sample_pred[i]}")

# Full Test Set Evaluation
print("\\n" + "="*50)
print("FULL TEST SET CLASSIFICATION REPORT")
print("="*50)

y_pred_full = best_model.predict(X_test)
print(classification_report(y_test, y_pred_full))

macro_f1 = f1_score(y_test, y_pred_full, average='macro')
print(f"\\nFINAL MACRO F1-SCORE ON TEST SET: {macro_f1:.4f}")
""")

with open('main_custom_lr.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Submission Notebook successfully generated at main_custom_lr.ipynb")
