import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
df_raw = pd.read_csv('./data/CEAS_08.csv', encoding='ISO-8859-1',
                     na_values=['na', 'NA', 'Unknown', ''])


# Clean column names
df_raw.columns = (
    df_raw.columns.str.strip()
    .str.replace(' ', '_')
    .str.replace(r'[^\w]', '', regex=True)
)

# Create a working copy and remove duplicates
df = df_raw.copy()

df.info()


before = len(df_raw)
df.drop_duplicates(inplace=True)
print(f"Removed {before - len(df)} duplicate rows.")

# Visualise missing values BEFORE cleaning
missing_before = df.isnull().sum().sort_values(ascending=False)
missing_df = missing_before.reset_index()
missing_df.columns = ['Column', 'MissingCount']

plt.figure(figsize=(10, 5))
sns.barplot(data=missing_df, x='Column', y='MissingCount', hue='Column', palette='Reds_r', legend=False)
plt.title("Missing Values Before Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize class distribution of label in pie chart
plt.figure(figsize=(8, 8))
label_counts = df['label'].value_counts()
colors = ['#ff9999', '#66b3ff']
explode = (0.05, 0)  # explode the first slice slightly

plt.pie(label_counts.values,
        labels=label_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True)
plt.title('Distribution of Email Labels (Phishing vs Legitimate)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

print(f"\nClass Distribution:")
print(label_counts)
print(f"\nClass Balance Ratio: {label_counts.min() / label_counts.max():.2%}")

# Handle Missing Values for receiver & subject
# Replace missing receivers (cannot drop these rows)
df['receiver'] = df['receiver'].fillna('unknown')
# Replace missing subjects (important for NLP)
# df['subject'] = df['subject'].fillna('no_subject')

# dropping rows with missing labels
# df = df.dropna(subset=['label'])

def extract_links(text):
    if pd.isna(text):
        return None
    urls = re.findall(r'(https?://[^\s]+)', str(text))
    return '; '.join(urls) if urls else None

# Handle links column properly
if 'links' in df.columns:
    df['links'] = df['links'].fillna(df['body'].apply(extract_links))
    # If still missing after extraction, fill with 'none'
    df['links'] = df['links'].fillna('none')
else:
    # If links column doesn't exist, create it from body
    df['links'] = df['body'].apply(extract_links)
    df['links'] = df['links'].fillna('none')


# def clean_text(text):
#     if pd.isna(text):
#         return ""
#     text = text.lower()
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'^(re|fwd):', '', text)
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#
#     return text.strip()

# df['body'] = df['body'].apply(clean_text)
# df['subject'] = df['subject'].apply(clean_text)



# Clean and standardize the date column

# Convert date strings to datetime (with UTC to handle mixed timezones)
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

# Fill missing dates with the most frequent valid date
mode_date = df['date'].mode(dropna=True)
if not mode_date.empty:
    df['date'] = df['date'].fillna(mode_date[0])
else:
    # If no valid dates exist, use a default date
    df['date'] = df['date'].fillna(pd.Timestamp('2000-01-01', tz='UTC'))

# Ensure the date column is datetime type
df['date'] = pd.to_datetime(df['date'], utc=True)

# Extract date features useful for ML
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['weekday'] = df['date'].dt.weekday

# Extract timezone offset
# df['tz_offset'] = df['date'].dt.strftime('%z')



#Visualise missing values AFTER cleaning
missing_after = df.isnull().sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_after.index, y=missing_after.values, hue=missing_after.index, palette='Greens', legend=False)
plt.title("Missing Values After Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Comparison chart of missing values before vs after
missing_comparison = pd.DataFrame({
    'Before Cleaning': missing_before,
    'After Cleaning': missing_after
})
missing_comparison.plot(kind='bar', figsize=(10, 5), color=['red', 'green'])
plt.title("Comparison of Missing Values Before and After Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Move label column to the end
if 'label' in df.columns:
    label_col = df['label']
    df = df.drop(columns=['label'])
    df['label'] = label_col


df.to_csv('./output/Cleaned_PhishingEmailData_with_dates.csv', index=False)

# Clean whitespaces and stray characters
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# Drop date for ML model
df_ml = df.drop(columns=['date'])
df_ml.to_csv('./output/Cleaned_PhishingEmailData.csv', index=False)


print("\nPreview of cleaned dataset:")
print(df.head())


