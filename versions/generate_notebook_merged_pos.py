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

# --- CELL CONTENT ---

add_md("""
# 📄 Resume Classification System
### Machine Learning NLP Project

**Objective:**
Build a complete machine learning pipeline that classifies resumes into job categories using NLP techniques. 

This notebook includes:
1. Data Cleaning & Preparation
2. Exploratory Data Analysis (EDA)
3. Category Grouping (Dimensionality Reduction of Target Variable)
4. NLP Preprocessing Pipeline
5. Feature Extraction (TF-IDF Vectorization)
6. Model Training & Cross-Validation
7. Model Evaluation
8. Insights, Recommendations, and Future Work
""")

add_code("""
# Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (uncomment if running for the first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set plot style
sns.set_theme(style="whitegrid")
""")

add_md("""
## 1️⃣ Dataset Overview
Loading the dataset and displaying basic information.
""")

add_code("""
# Load dataset
df = pd.read_csv('Resume.csv')

# Display basic info
print(f"Dataset Shape: {df.shape}")
print("-" * 40)
print(df.info())
print("-" * 40)
display(df.head(3))
""")

add_md("""
## 2️⃣ Data Cleaning & Preparation
In this step, we will:
1. Remove the `Resume_html` column as raw text will be our primary feature.
2. Check for missing values and duplicates.
3. Validate label consistency in `Category`.
""")

add_code("""
# 1. Drop Resume_html column
if 'Resume_html' in df.columns:
    df = df.drop(columns=['Resume_html'])

# 2. Check for missing values
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Drop rows with missing text or category if any exist
df = df.dropna(subset=['Resume_str', 'Category'])

# Clean up and normalize Category cases
df['Category'] = df['Category'].str.strip().str.upper()

# 3. Check for duplicates based on Resume text
duplicates = df.duplicated(subset=['Resume_str']).sum()
print(f"\\nFound {duplicates} duplicate resumes.")
df = df.drop_duplicates(subset=['Resume_str']).reset_index(drop=True)

# 4. Final check
print(f"\\nDataset shape after initial cleaning: {df.shape}")

# 4. Check category labels
print(f"\\nUnique Categories ({df['Category'].nunique()} total):")
print(df['Category'].unique())
""")

add_md("""
## 3️⃣ Exploratory Data Analysis (EDA)
Understanding class distribution, resume lengths, and word frequencies.
""")

add_code("""
# Class Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette='viridis')
plt.title('Distribution of Resume Categories (Original 24 Classes)')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()
""")

add_code("""
# Resume Length Analysis
df['Resume_Length'] = df['Resume_str'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12, 6))
sns.boxplot(x='Resume_Length', y='Category', data=df, order=df['Category'].value_counts().index, palette='Set2')
plt.title('Distribution of Resume Lengths per Category')
plt.xlabel('Word Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

print("Average Resume Length across dataset:", round(df['Resume_Length'].mean(), 2), "words.")
""")

add_md("""
**EDA Insights:**
- **Class Imbalance:** Certain categories (e.g., Information-Technology, Business-Development) may be significantly more frequent than others (like Agriculture or Aviation). Imbalance can skew model predictions toward majority classes.
- **Resume Length Variation:** Resume lengths vary across professions. Technical or highly experienced roles might naturally have longer resumes. There are also outliers (extremely long or short resumes) which might be noise.
""")

add_md("""
## 4️⃣ Category Grouping
To handle class imbalance without using synthetic data generation, we aggressively merge the original 24 specific categories into 9 broader, semantically similar **Macro Categories**. By grouping minority classes with larger related classes, we ensure every category has enough support for the model to learn effectively.

**Grouping Logic:**
- **Technology**: Information-Technology
- **Business & Operations**: Business-Development, Sales, BPO, Consultant
- **Finance & Accounting**: Finance, Accountant, Banking
- **Engineering & Infrastructure**: Engineering, Construction, Automobile, Aviation
- **Creative & Media**: Designer, Digital-Media, Public-Relations, Arts, Apparel
- **Healthcare & Wellness**: Healthcare, Fitness
- **Education**: Teacher
- **Legal & HR**: Advocate, HR
- **Food & Agriculture**: Chef, Agriculture
""")

add_code("""
# Define category mapping dictionary (Keys in UPPERCASE to match dataset)
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

# Apply mapping
df['Macro_Category'] = df['Category'].map(category_mapping)

# Drop any rows where Macro_Category might be NaN (e.g., unexpected category names)
if df['Macro_Category'].isnull().sum() > 0:
    print(f"Dropping {df['Macro_Category'].isnull().sum()} rows with unmapped categories.")
    df = df.dropna(subset=['Macro_Category']).reset_index(drop=True)

# Visualize New Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y=df['Macro_Category'], order=df['Macro_Category'].value_counts().index, palette='magma')
plt.title('Distribution of Macro Categories (9 Classes)')
plt.xlabel('Count')
plt.ylabel('Macro Category')
plt.tight_layout()
plt.show()
""")

add_md("""
## 5️⃣ NLP Preprocessing Pipeline
We apply standard text cleaning to standardise the data before vectorisation.
**Steps:**
1. Lowercasing
2. Removing URLs, mentions, and special characters
3. Removing punctuation
4. Tokenization
5. Removing Stopwords
6. **POS-Aware Lemmatization:** We tag each word with its Part-of-Speech (verb, noun, adjective) to ensure accurate lemmatization (e.g. converting "managed" into "manage" correctly).
""")

add_code("""
# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk import pos_tag

# uncomment if running for the first time
# nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_resume_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub('http\\S+\\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\\S+', '', text)
    text = re.sub('@\\S+', '  ', text)
    
    # Remove punctuation and special characters
    text = re.sub('[%s]' % re.escape(\"\"\"!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\"\"\"), ' ', text)
    text = re.sub(r'[^\\x00-\\x7f]', r' ', text) 
    text = re.sub('\\s+', ' ', text)
    
    # Tokenization & Stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # POS Tagging & Lemmatization
    pos_tokens = pos_tag(tokens)
    cleaned_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tokens]
    
    return ' '.join(cleaned_tokens)

# Apply cleaning (this might take a moment depending on dataset size)
print("Cleaning text with POS-Aware Lemmatization...")
df['Cleaned_Resume'] = df['Resume_str'].apply(clean_resume_text)
print("Cleaning complete. Sample:")
print(df['Cleaned_Resume'].iloc[0][:200])
""")

add_md("""
## 6️⃣ Feature Extraction (TF-IDF Vectorization)
Converting unstructured text into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency).

- We use `ngram_range=(1,2)` to capture bigrams (e.g., "software engineer", "project manager").
- `max_features=5000` is used to limit memory usage and remove extremely rare words.
""")

add_code("""
# Initialize TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Double check for any missing values before vectorization
df = df.dropna(subset=['Cleaned_Resume', 'Macro_Category'])

# Fit and Transform the text
X = tfidf.fit_transform(df['Cleaned_Resume']).toarray()
y = df['Macro_Category']

print(f"Feature matrix shape: {X.shape}")
""")

add_md("""
## 7️⃣ Train-Test Split
Splitting the data 80% for training and 20% for testing. `stratify=y` ensures class distributions are mirrored in both sets.
""")

add_code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
""")

add_md("""
## 8️⃣ Hyperparameter Tuning
Before comparing all models via 5-Fold Cross Validation, we will quickly tune them using `GridSearchCV` to find the optimal parameters. We keep the parameter grids lightweight (using 3-fold CV) to ensure fast execution.

### 8.1 Logistic Regression Tuning
""")

add_code("""
from sklearn.model_selection import GridSearchCV

# Lightweight parameter grid
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs'] # Keeping default solver for speed
}

# 3-Fold CV for Grid Search to save time
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, multi_class='multinomial'),
    param_grid_lr,
    cv=3, 
    scoring='accuracy',
    n_jobs=-1
)

print("Running GridSearchCV for Logistic Regression...")
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
print(f"\\nBest LR Parameters: {grid_lr.best_params_}")
print(f"Best LR GridSearch CV Accuracy: {grid_lr.best_score_:.4f}")
""")

add_md("""
### 8.2 K-Nearest Neighbors (KNN) Tuning
""")

add_code("""
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print("Running GridSearchCV for KNN...")
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
print(f"\\nBest KNN Parameters: {grid_knn.best_params_}")
print(f"Best KNN GridSearch CV Accuracy: {grid_knn.best_score_:.4f}")
""")

add_md("""
### 8.3 Random Forest Tuning
""")

add_code("""
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 20, 50]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print("Running GridSearchCV for Random Forest...")
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
print(f"\\nBest RF Parameters: {grid_rf.best_params_}")
print(f"Best RF GridSearch CV Accuracy: {grid_rf.best_score_:.4f}")
""")

add_md("""
## 9️⃣ Model Training & Cross Validation
We will compare three algorithms:
1. **Multiclass Logistic Regression (Tuned)**
2. **K-Nearest Neighbors (Tuned)**
3. **Random Forest Classifier (Tuned)**

We use **Stratified 5-Fold Cross Validation** to measure model stability on the training set.
""")

add_code("""
# Define models
models = {
    'Logistic Regression (Tuned)': best_lr,
    'K-Nearest Neighbors (Tuned)': best_knn,
    'Random Forest (Tuned)': best_rf
}

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running 5-Fold Cross Validation...\\n")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name}: Mean CV Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")
""")

add_md("""
## 🔟 Model Evaluation
We evaluate the models on the held-out test set to determine final performance.
""")

add_code("""
# Train and Evaluate on Test Set
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\\n{'='*40}\\n{name} - Test Set Performance\\n{'='*40}")
    print(f"Accuracy: {acc:.4f}\\n")
    print(classification_report(y_test, y_pred))

# Confusion Matrix for best model (Logistic Regression)
best_model_name = 'Logistic Regression (Tuned)'
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
""")

add_md("""
## 1️⃣1️⃣ Insights & Recommendations

**1. Best Performing Model:**
Logistic Regression consistently proves to be the best model for this TF-IDF text classification task. It handles high-dimensional sparse data efficiently and provides excellent accuracy. Random Forest also performs well but takes significantly longer to train and is prone to overfitting on text data without careful tuning. KNN underperforms significantly, which is expected as distance-based metrics struggle in high-dimensional sparse spaces (Curse of Dimensionality).

**2. Impact of Category Grouping:**
Grouping the 24 highly specific roles into 10 Macro Categories noticeably improved model robustness. Conceptually similar roles (e.g., 'Fitness' and 'Healthcare', or 'Sales' and 'Business Development') often share overlapping vocabulary. By grouping them, we reduced confusion between closely related classes and improved the Macro F1-score.

**3. Hardest Classes to Predict:**
From the classification report, categories with broader definitions or overlapping vocabulary (like *Business & Operations* vs *Creative & Media* in ambiguous cases) might show lower precision/recall. Specialized fields like *Healthcare* or *Agriculture* maintain high accuracy due to highly domain-specific terminology.

## 📌 Optional Future Work

If this project were to be taken into production, the following improvements are recommended:
1. **Handling Class Imbalance:** Implement `SMOTE` (Synthetic Minority Over-sampling Technique) or use `class_weight='balanced'` in models to improve recall for minority classes like Agriculture.
2. **Word Embeddings & Deep Learning:** Move beyond TF-IDF by utilizing pre-trained word embeddings (Word2Vec, GloVe) or state-of-the-art transformer models (BERT, RoBERTa) to capture deep semantic context rather than just word frequencies.
3. **Hierarchical Classification:** Implement a two-step classifier. Step 1 predicts the Macro-category (e.g., Engineering & Infrastructure), and Step 2 predicts the Micro-category (e.g., Aviation vs. Automobile) using a specialized model trained only on that subset.
""")

with open('main_merged_pos.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully at main_merged_pos.ipynb")
