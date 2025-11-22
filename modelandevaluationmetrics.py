# Step 0: Suppress TensorFlow and general warnings to keep output clean
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow info/warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimizations for consistent results

import warnings
warnings.filterwarnings("ignore") # Suppress all Python warnings


import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations
import re # For regular expressions used in text cleaning



from sklearn.model_selection import train_test_split # For splitting data
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.linear_model import LogisticRegression # Logistic Regression model
from sklearn.naive_bayes import GaussianNB # Naive Bayes model
from xgboost import XGBClassifier # XGBoost model
from sklearn.preprocessing import StandardScaler # For feature scaling
from sklearn.metrics import classification_report, accuracy_score, precision_score,recall_score, f1_score, roc_auc_score # Evaluation metrics


from tensorflow.keras.models import Sequential # Sequential model builder
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
#Layers for LSTM model
from tensorflow.keras.preprocessing.text import Tokenizer # Tokenizer for textpreprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences # Padding sequences to equal length


from transformers import BertTokenizer, TFBertForSequenceClassification # BERT tokenizer and model
from transformers import logging
import tensorflow as tf
logging.set_verbosity_error() # Suppress transformer warnings

df = pd.read_csv('Cleaned_PhishingEmailData.csv', encoding='ISO-8859-1') # Load CSV with proper encoding
# Step 6: Feature engineering from email content

def clean_text(text):
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r"http\S+", "link", text) # Replace URLs with 'link'
    text = re.sub(r"\S+@\S+", "email", text) # Replace email addresses with 'email'
    text = re.sub(r"[^a-z\s]", "", text) # Remove non-alphabetic characters
    return text
# Apply text cleaning and extract features
df['clean_content'] = df['Email_Content'].apply(clean_text)
df['subject_length'] = df['Email_Subject'].apply(lambda x: len(str(x))) # Length of subject line
df['link_count'] = df['Email_Content'].apply(lambda x: len(re.findall(r"http\S+", str(x)))) # Number of links
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0) # Binary indicator for links