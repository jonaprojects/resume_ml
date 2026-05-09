import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('Resume.csv')
df = df.dropna(subset=['Resume_str', 'Category'])
df['Category'] = df['Category'].str.strip().str.upper()

# Base mapping common to all
base_mapping = {
    'INFORMATION-TECHNOLOGY': 'Technology',
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
    'ADVOCATE': 'Legal',
    'HR': 'Human Resources',
    'CHEF': 'Food & Hospitality',
    'AGRICULTURE': 'Agriculture'
}

def evaluate_pipeline(name, mapping, use_class_weight=False, use_smote=False):
    # Apply mapping
    temp_df = df.copy()
    temp_df['Macro_Category'] = temp_df['Category'].map(mapping)
    temp_df = temp_df.dropna(subset=['Macro_Category'])
    
    # Fast TF-IDF directly on raw text
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(temp_df['Resume_str'])
    y = temp_df['Macro_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if use_smote:
        # SMOTE k_neighbors must be smaller than the smallest class size
        min_class_size = y_train.value_counts().min()
        k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    clf = LogisticRegression(max_iter=1000, class_weight='balanced' if use_class_weight else None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return acc, macro_f1

# MAPPING A: Merged
mapping_merged = base_mapping.copy()
mapping_merged.update({
    'BUSINESS-DEVELOPMENT': 'Sales & Business Dev',
    'SALES': 'Sales & Business Dev',
    'BPO': 'Sales & Business Dev',
    'CONSULTANT': 'Sales & Business Dev'
})

# MAPPING B: Separated
mapping_sep = base_mapping.copy()
mapping_sep.update({
    'BUSINESS-DEVELOPMENT': 'Sales & Business Dev',
    'SALES': 'Sales & Business Dev',
    'BPO': 'BPO & Operations',
    'CONSULTANT': 'Consulting'
})

print("Testing scenarios...")
acc_A, f1_A = evaluate_pipeline("Merged (No Balancing)", mapping_merged)
print(f"Scenario A (Merged, No Balancing): Accuracy = {acc_A:.4f}, Macro F1 = {f1_A:.4f}")

acc_B, f1_B = evaluate_pipeline("Separated + Class Weight", mapping_sep, use_class_weight=True)
print(f"Scenario B (Separated + Class Weights): Accuracy = {acc_B:.4f}, Macro F1 = {f1_B:.4f}")

try:
    acc_C, f1_C = evaluate_pipeline("Separated + SMOTE", mapping_sep, use_smote=True)
    print(f"Scenario C (Separated + SMOTE): Accuracy = {acc_C:.4f}, Macro F1 = {f1_C:.4f}")
except Exception as e:
    print(f"Scenario C (Separated + SMOTE) failed: {e}")
