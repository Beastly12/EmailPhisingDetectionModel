
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# -------------------------
# 1. Load dataset
# -------------------------
INPUT_CSV = './Cleaned_PhishingEmailData.csv'  # adjust if needed
df = pd.read_csv(INPUT_CSV, encoding='ISO-8859-1')
print("Dataset loaded. Shape:", df.shape)

# Print columns for quick inspection
print("Columns:", df.columns.tolist())

# -------------------------
# 2. Find the Label column (robust)
# -------------------------
# Accept common variations if 'Label' doesn't exist
possible_label_names = ['Label', 'label', 'Target', 'target', 'Class', 'class', 'is_phishing', 'Is_Phishing']
label_col = None
for name in possible_label_names:
    if name in df.columns:
        label_col = name
        break

# If not found, try to guess by looking for a low-cardinality object column
if label_col is None:
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 10:
            # candidate label (warn user)
            label_col = col
            print(f"Guessing label column as '{col}' (dtype object, nunique <= 10).")
            break

if label_col is None:
    raise ValueError("No label column found automatically. Please set variable 'label_col' to your label column name.")

print("Using label column:", label_col)

# -------------------------
# 3. Basic EDA: class distribution
# -------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=label_col, data=df, palette='Set2')
plt.title("Label Distribution")
plt.show()

print("\nLabel counts (absolute):")
print(df[label_col].value_counts())
print("\nLabel distribution (percent):")
print(df[label_col].value_counts(normalize=True) * 100)

# -------------------------
# 4. Create basic length features
# -------------------------
# Safely reference columns — adapt if your dataset uses different names:
subject_col = 'Email_Subject' if 'Email_Subject' in df.columns else ('Subject' if 'Subject' in df.columns else None)
body_col = 'Body' if 'Body' in df.columns else ('EmailBody' if 'EmailBody' in df.columns else None)
links_col = 'Links' if 'Links' in df.columns else None
sender_col = 'Sender_Email' if 'Sender_Email' in df.columns else ('From' if 'From' in df.columns else None)

if subject_col:
    df['subject_length'] = df[subject_col].astype(str).apply(len)
else:
    print("Warning: Subject column not found. 'subject_length' skipped.")

if body_col:
    df['body_length'] = df[body_col].astype(str).apply(len)
else:
    print("Warning: Body column not found. 'body_length' skipped.")

# -------------------------
# 5. Number of links feature (with extraction fallback)
# -------------------------
def extract_links_from_text(text):
    if pd.isna(text) or text == '':
        return []
    # simple URL regex
    return re.findall(r'https?://[^\s\'"<>)]+', str(text))

if links_col and links_col in df.columns:
    # If links column exists, count semicolon-separated or extracted URLs if 'none'
    def count_links_cell(x):
        if pd.isna(x) or str(x).strip().lower() in ('none', ''):
            return 0
        # split on semicolon, comma, or whitespace fallback
        parts = re.split(r'[;,]\s*', str(x))
        # filter empties and non-urls by basic check
        return sum(1 for p in parts if re.search(r'https?://', p))
    df['num_links'] = df[links_col].apply(count_links_cell)
else:
    # fallback: try to extract links from body text
    if body_col:
        df['num_links'] = df[body_col].apply(lambda t: len(extract_links_from_text(t)))
    else:
        df['num_links'] = 0
        print("Warning: No Links or Body column found — num_links set to 0.")

# -------------------------
# 6. Domains from Links
# -------------------------
def get_domains(link_string, body_fallback=None):
    domains = []
    if pd.isna(link_string) or str(link_string).strip().lower() in ('none', ''):
        # try fallback (body_fallback is raw body text)
        if body_fallback:
            urls = extract_links_from_text(body_fallback)
        else:
            urls = []
    else:
        # split semicolon/comma sep and extract host
        parts = re.split(r'[;,]\s*', str(link_string))
        urls = []
        for p in parts:
            match = re.search(r'https?://([^/:\s]+)', p)
            if match:
                urls.append(match.group(0))  # full host e.g. 'example.com'
        # if none found, fallback to body
        if not urls and body_fallback:
            urls = extract_links_from_text(body_fallback)
    # extract domain host from urls
    for u in urls:
        m = re.search(r'https?://([^/:\s]+)', u)
        if m:
            domains.append(m.group(1).lower())
    return domains

df['domains'] = df.apply(lambda r: get_domains(r[links_col], body_fallback=(r[body_col] if body_col in r else None))
                         if links_col in df.columns or body_col in df.columns else [], axis=1)

# Top domains (print)
all_domains = Counter([domain for sublist in df['domains'] for domain in sublist])
print("\nTop domains in links (most common 20):")
print(all_domains.most_common(20))

# -------------------------
# 7. Keyword count features
# -------------------------
suspicious_words = [
    'verify','urgent','account','password','click','alert','login','bank',
    'security','update','confirm','suspend','billing','payment','unlock'
]
def keyword_count(text):
    s = str(text).lower()
    return sum(1 for word in suspicious_words if word in s)

if subject_col:
    df['subject_keyword_count'] = df[subject_col].apply(keyword_count)
else:
    df['subject_keyword_count'] = 0

if body_col:
    df['body_keyword_count'] = df[body_col].apply(keyword_count)
else:
    df['body_keyword_count'] = 0

# -------------------------
# 8. Sender domain and public-sender flag
# -------------------------
if sender_col and sender_col in df.columns:
    df['sender_domain'] = df[sender_col].astype(str).apply(lambda x: x.split('@')[-1].lower() if '@' in x else x.lower())
else:
    df['sender_domain'] = 'unknown'

public_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}
df['sender_is_public'] = df['sender_domain'].apply(lambda x: x in public_domains)

# -------------------------
# 9. Text-shape features
# -------------------------
if body_col:
    df['num_exclamations'] = df[body_col].astype(str).apply(lambda x: x.count('!'))
    df['num_uppercase_words'] = df[body_col].astype(str).apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    df['num_numbers'] = df[body_col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)))
else:
    df['num_exclamations'] = 0
    df['num_uppercase_words'] = 0
    df['num_numbers'] = 0

# -------------------------
# 10. Label encoding (do NOT remove original label)
# -------------------------
# If label is textual (e.g., 'phishing'/'legitimate'), make a numeric copy for correlation/modeling
if df[label_col].dtype == object:
    mapped = None
    # common mappings
    if set(df[label_col].str.lower().unique()) >= {'phishing','legitimate'} or set(df[label_col].str.lower().unique()) <= {'phishing','legitimate'}:
        mapped = df[label_col].str.lower().map({'phishing': 1, 'legitimate': 0})
    elif set(df[label_col].str.lower().unique()) >= {'spam','ham'} or set(df[label_col].str.lower().unique()) <= {'spam','ham'}:
        mapped = df[label_col].str.lower().map({'spam': 1, 'ham': 0})
    else:
        # fallback: convert categories to codes
        mapped = pd.Categorical(df[label_col]).codes
    df['_label_num'] = mapped
else:
    df['_label_num'] = df[label_col].astype(int)

print("\n_label_num value counts:")
print(df['_label_num'].value_counts())

# -------------------------
# 11. Prepare numeric dataframe for correlation
# -------------------------
# Choose numeric columns we engineered plus the numeric label
numeric_cols = [
    'subject_length','body_length','num_links','subject_keyword_count','body_keyword_count',
    'sender_is_public','num_exclamations','num_uppercase_words','num_numbers'
]
# keep only those present
numeric_cols = [c for c in numeric_cols if c in df.columns]
numeric_cols = numeric_cols + ['_label_num']

numeric_df = df[numeric_cols].copy()
print("\nNumeric columns used for correlation:", numeric_df.columns.tolist())

# Fill NA numeric with 0 (safe for correlation display)
numeric_df = numeric_df.fillna(0)

# -------------------------
# 12. Correlation matrix & heatmap
# -------------------------
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4)
plt.title("Feature Correlation Heatmap")
plt.show()

# Show correlations with label sorted
label_corr = corr_matrix['_label_num'].sort_values(ascending=False)
print("\nCorrelation of features with label (descending):")
print(label_corr)

# -------------------------
# 13. Optional: Feature distributions for each numeric feature
# -------------------------
for col in [c for c in numeric_cols if c != '_label_num']:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x=col, hue=label_col, common_norm=False)
    plt.title(f"Distribution of {col} by {label_col}")
    plt.show()

# -------------------------
# 14. Save engineered dataset (optional)
# -------------------------
OUT_CSV = './engineered_phishing_dataset_fixed.csv'
df.to_csv(OUT_CSV, index=False)
print("Saved engineered dataset to:", OUT_CSV)
