"""
FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS
================================================
This script performs:
1. Feature Engineering: Extract metadata and create new variables
2. Privacy Preservation: Anonymize sensitive information
3. Exploratory Data Analysis: Visualize patterns and distributions
"""

# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import hashlib
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ============================================
# STEP 1: LOAD CLEANED DATA
# ============================================
print("\n[STEP 1] Loading cleaned dataset...")
df = pd.read_csv('./data/Cleaned_PhishingEmailData.csv')

print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['label'].value_counts())

# ============================================
# STEP 2: DOMAIN EXTRACTION
# ============================================
print("\n" + "=" * 70)
print("[STEP 2] EXTRACTING DOMAIN FEATURES")
print("=" * 70)


def extract_domain(email):
    """Extract domain from email address"""
    if pd.isna(email) or email == 'unknown':
        return 'unknown'
    try:
        return email.split('@')[-1].lower()
    except:
        return 'unknown'


print("Extracting sender and receiver domains...")
df['sender_domain'] = df['sender'].apply(extract_domain)
df['receiver_domain'] = df['receiver'].apply(extract_domain)

print(f"✓ Unique sender domains: {df['sender_domain'].nunique()}")
print(f"✓ Unique receiver domains: {df['receiver_domain'].nunique()}")

# ============================================
# STEP 3: EMAIL ANONYMIZATION (Privacy Compliance)
# ============================================
print("\n" + "=" * 70)
print("[STEP 3] ANONYMIZING EMAIL ADDRESSES (Privacy Preservation)")
print("=" * 70)


def anonymise_email(email):
    """Hash email addresses using SHA-256 for privacy"""
    if pd.isna(email) or email == 'unknown':
        return 'unknown'
    return hashlib.sha256(email.encode()).hexdigest()


print("Hashing email addresses with SHA-256...")
df['sender_hash'] = df['sender'].apply(anonymise_email)
df['receiver_hash'] = df['receiver'].apply(anonymise_email)

print("✓ Email addresses anonymized")
print(f"  Example original: {df['sender'].iloc[0]}")
print(f"  Example hashed: {df['sender_hash'].iloc[0][:16]}...")

# Drop original email columns for privacy compliance
print("\nDropping original email columns for privacy compliance...")
df = df.drop(columns=['sender', 'receiver'])
print("✓ Original email addresses removed from dataset")

# ============================================
# STEP 4: SUBJECT-BASED FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 4] EXTRACTING SUBJECT-BASED FEATURES")
print("=" * 70)


def safe_len(text):
    """Safely get length of text"""
    if pd.isna(text):
        return 0
    return len(str(text))


print("Creating subject metadata features...")
df['subject_length'] = df['subject'].apply(safe_len)
df['subject_num_special'] = df['subject'].fillna('').str.count(r'[^a-zA-Z0-9 ]')
df['subject_upper_words'] = df['subject'].fillna('').apply(
    lambda x: sum(1 for w in str(x).split() if w.isupper())
)
df['subject_has_re_fwd'] = df['subject'].fillna('').str.contains(
    r'\b(re|fwd)\b', case=False, regex=True
).astype(int)

print(f"✓ Average subject length: {df['subject_length'].mean():.2f} characters")
print(f"✓ Emails with RE/FWD: {df['subject_has_re_fwd'].sum()} ({df['subject_has_re_fwd'].mean() * 100:.1f}%)")

# ============================================
# STEP 5: BODY-BASED FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 5] EXTRACTING BODY-BASED FEATURES")
print("=" * 70)

print("Creating body metadata features...")
df['body_length'] = df['body'].apply(safe_len)
df['body_num_digits'] = df['body'].fillna('').str.count(r'\d')
df['body_num_exclaim'] = df['body'].fillna('').str.count('!')
df['body_num_question'] = df['body'].fillna('').str.count(r'\?')
df['body_percent_caps'] = df['body'].fillna('').apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

# Check for HTML content (if tags exist in original)
if 'contains_html' not in df.columns:
    df['contains_html'] = df['body'].fillna('').str.contains('<[^>]+>', regex=True).astype(int)

print(f"✓ Average body length: {df['body_length'].mean():.2f} characters")
print(f"✓ Average exclamation marks: {df['body_num_exclaim'].mean():.2f}")
print(f"✓ Emails with HTML: {df['contains_html'].sum()} ({df['contains_html'].mean() * 100:.1f}%)")

# ============================================
# STEP 6: URL-BASED FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 6] EXTRACTING URL-BASED FEATURES")
print("=" * 70)

print("Analyzing URL patterns...")
# Count URLs from the 'links' column
df['num_urls'] = df['links'].apply(lambda x: 0 if x == 'none' else len(str(x).split(';')))

# Check for suspicious URL patterns (URL shorteners)
df['has_suspicious_url'] = df['links'].fillna('').str.contains(
    r'(bit\.ly|tinyurl|shorturl|goo\.gl|t\.co)', case=False, regex=True
).astype(int)

print(f"✓ Average URLs per email: {df['num_urls'].mean():.2f}")
print(f"✓ Emails with suspicious URLs: {df['has_suspicious_url'].sum()} ({df['has_suspicious_url'].mean() * 100:.1f}%)")

# ============================================
# STEP 7: TEMPORAL FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 7] CREATING TEMPORAL FEATURES")
print("=" * 70)

# Add is_weekend if not already present
if 'is_weekend' not in df.columns and 'weekday' in df.columns:
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    print(f"✓ Weekend emails: {df['is_weekend'].sum()} ({df['is_weekend'].mean() * 100:.1f}%)")

# Add time of day categories
if 'hour' in df.columns:
    def categorize_time(hour):
        if pd.isna(hour):
            return 'unknown'
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'


    df['time_of_day'] = df['hour'].apply(categorize_time)
    print(f"✓ Time of day distribution:")
    print(df['time_of_day'].value_counts())

# ============================================
# STEP 8: INTERACTION & DERIVED FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 8] CREATING INTERACTION FEATURES")
print("=" * 70)

print("Creating derived features...")
# Ratio of subject to body length
df['subject_body_ratio'] = df['subject_length'] / (df['body_length'] + 1)

# Urgency indicators (common in phishing emails)
urgency_keywords = ['urgent', 'immediate', 'action required', 'verify',
                    'suspended', 'expired', 'confirm', 'alert']
df['urgency_score'] = df['subject'].fillna('').str.lower().apply(
    lambda x: sum(keyword in x for keyword in urgency_keywords)
)

# Financial indicators
financial_keywords = ['bank', 'account', 'payment', 'credit', 'paypal',
                      'transaction', 'invoice', 'refund']
df['financial_score'] = df['body'].fillna('').str.lower().apply(
    lambda x: sum(keyword in x for keyword in financial_keywords)
)

print(f"✓ Emails with urgency indicators: {(df['urgency_score'] > 0).sum()}")
print(f"✓ Emails with financial keywords: {(df['financial_score'] > 0).sum()}")

# ============================================
# STEP 9: EXPLORATORY DATA ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("[STEP 9] EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Create output directory for plots
import os

os.makedirs('outputs/eda_plots', exist_ok=True)

# 9.1: Class Distribution
print("\n[9.1] Analyzing class distribution...")
plt.figure(figsize=(8, 5))
df['label'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Class Distribution: Phishing vs Legitimate Emails', fontsize=14, fontweight='bold')
plt.xlabel('Email Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_plots/01_class_distribution.png', dpi=300)
plt.show()

class_dist = df['label'].value_counts(normalize=True) * 100
print(f"✓ Class balance:")
for label, pct in class_dist.items():
    print(f"  {label}: {pct:.2f}%")

# 9.2: Subject Length Distribution by Class
print("\n[9.2] Analyzing subject length patterns...")
plt.figure(figsize=(12, 5))
df.boxplot(column='subject_length', by='label', patch_artist=True)
plt.suptitle('')
plt.title('Subject Length Distribution by Email Type', fontsize=14, fontweight='bold')
plt.xlabel('Email Type', fontsize=12)
plt.ylabel('Subject Length (characters)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/eda_plots/02_subject_length_distribution.png', dpi=300)
plt.show()

print(f"✓ Average subject length by class:")
print(df.groupby('label')['subject_length'].mean())

# 9.3: Body Length Distribution by Class
print("\n[9.3] Analyzing body length patterns...")
plt.figure(figsize=(12, 5))
df.boxplot(column='body_length', by='label', patch_artist=True)
plt.suptitle('')
plt.title('Body Length Distribution by Email Type', fontsize=14, fontweight='bold')
plt.xlabel('Email Type', fontsize=12)
plt.ylabel('Body Length (characters)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/eda_plots/03_body_length_distribution.png', dpi=300)
plt.show()

print(f"✓ Average body length by class:")
print(df.groupby('label')['body_length'].mean())

# 9.4: URL Count Distribution
print("\n[9.4] Analyzing URL patterns...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df.groupby('label')['num_urls'].mean().plot(kind='bar', color=['#3498db', '#e67e22'])
plt.title('Average Number of URLs by Email Type', fontsize=12, fontweight='bold')
plt.xlabel('Email Type', fontsize=11)
plt.ylabel('Average URL Count', fontsize=11)
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df.groupby('label')['has_suspicious_url'].sum().plot(kind='bar', color=['#9b59b6', '#f39c12'])
plt.title('Suspicious URLs by Email Type', fontsize=12, fontweight='bold')
plt.xlabel('Email Type', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('outputs/eda_plots/04_url_analysis.png', dpi=300)
plt.show()

# 9.5: Temporal Patterns - Hour Distribution
if 'hour' in df.columns:
    print("\n[9.5] Analyzing temporal patterns...")
    plt.figure(figsize=(14, 5))

    for i, label in enumerate(df['label'].unique()):
        plt.subplot(1, 2, i + 1)
        df[df['label'] == label]['hour'].value_counts().sort_index().plot(
            kind='bar', color='#3498db' if i == 0 else '#e74c3c'
        )
        plt.title(f'Hour Distribution: {label} Emails', fontsize=12, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/eda_plots/05_temporal_patterns.png', dpi=300)
    plt.show()

# 9.6: Weekend vs Weekday Analysis
if 'is_weekend' in df.columns:
    print("\n[9.6] Analyzing weekend vs weekday patterns...")
    plt.figure(figsize=(10, 5))

    weekend_dist = pd.crosstab(df['is_weekend'], df['label'], normalize='columns') * 100
    weekend_dist.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
    plt.title('Weekend vs Weekday Email Distribution by Type', fontsize=14, fontweight='bold')
    plt.xlabel('Day Type (0=Weekday, 1=Weekend)', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Email Type')
    plt.tight_layout()
    plt.savefig('outputs/eda_plots/06_weekend_analysis.png', dpi=300)
    plt.show()

# 9.7: Urgency and Financial Keywords
print("\n[9.7] Analyzing urgency and financial indicators...")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
df.groupby('label')['urgency_score'].mean().plot(kind='bar', color=['#e74c3c', '#95a5a6'])
plt.title('Average Urgency Score by Email Type', fontsize=12, fontweight='bold')
plt.xlabel('Email Type', fontsize=11)
plt.ylabel('Average Urgency Score', fontsize=11)
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df.groupby('label')['financial_score'].mean().plot(kind='bar', color=['#f39c12', '#95a5a6'])
plt.title('Average Financial Keywords by Email Type', fontsize=12, fontweight='bold')
plt.xlabel('Email Type', fontsize=11)
plt.ylabel('Average Financial Score', fontsize=11)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('outputs/eda_plots/07_keyword_analysis.png', dpi=300)
plt.show()

# 9.8: Top Sender Domains
print("\n[9.8] Analyzing top sender domains...")
plt.figure(figsize=(12, 6))

top_domains = df['sender_domain'].value_counts().head(15)
top_domains.plot(kind='barh', color='#3498db')
plt.title('Top 15 Sender Domains', fontsize=14, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Domain', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/eda_plots/08_top_domains.png', dpi=300)
plt.show()

print(f"✓ Top 5 sender domains:")
print(top_domains.head())

# 9.9: Correlation Heatmap of Numerical Features
print("\n[9.9] Creating correlation heatmap...")
numerical_features = ['subject_length', 'body_length', 'num_urls', 'urgency_score',
                      'financial_score', 'subject_num_special', 'body_num_exclaim',
                      'body_num_digits', 'body_percent_caps']

# Filter only existing columns
numerical_features = [col for col in numerical_features if col in df.columns]

plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_plots/09_correlation_heatmap.png', dpi=300)
plt.show()

# 9.10: Feature Summary Statistics
print("\n[9.10] Generating feature summary statistics...")
summary_stats = df[numerical_features + ['label']].groupby('label').agg(['mean', 'std', 'min', 'max'])
print("\n✓ Summary Statistics by Email Type:")
print(summary_stats)

# Save summary statistics
summary_stats.to_csv('outputs/eda_plots/feature_summary_statistics.csv')
print("✓ Saved summary statistics to: outputs/eda_plots/feature_summary_statistics.csv")

# ============================================
# STEP 10: SAVE ENGINEERED FEATURES
# ============================================
print("\n" + "=" * 70)
print("[STEP 10] SAVING ENGINEERED FEATURES")
print("=" * 70)

print(f"Total features created: {len(df.columns)}")
print(f"Total samples: {len(df)}")

# Move label to the end
if 'label' in df.columns:
    label_col = df['label']
    df = df.drop(columns=['label'])
    df['label'] = label_col

# Save engineered dataset
df.to_csv('outputs/Engineered_Features.csv', index=False)
print("\n✓ Saved engineered features to: outputs/Engineered_Features.csv")

# Preview final dataset
print("\nPreview of final engineered dataset:")
print(df.head())

print("\n" + "=" * 70)
print("FEATURE ENGINEERING & EDA COMPLETE!")
print("=" * 70)
print(f"\n✓ Total features: {len(df.columns)}")
print(f"✓ Total samples: {len(df)}")
print(f"✓ Output files generated:")
print(f"  - outputs/Engineered_Features.csv")
print(f"  - outputs/eda_plots/ (9 visualization files)")
print(f"  - outputs/eda_plots/feature_summary_statistics.csv")
print("\nNext steps:")
print("  1. Review the EDA visualizations in outputs/eda_plots/")
print("  2. Run clustering analysis (clustering.py)")
print("  3. Build machine learning models (ml_models.py)")
