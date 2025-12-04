import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import pickle
import warnings

warnings.filterwarnings('ignore')

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

output_dir = 'outputs/ml_models'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("PHISHING EMAIL DETECTION - ML MODELS")
print("=" * 80)

# Load the data
df = pd.read_csv('outputs/Engineered_Features.csv')
print(f"\nData loaded: {len(df)} emails, {len(df.columns)} columns")

# Split features and target
y = df['label']
X = df.drop(columns=['label'])

# Check what we're working with
print("\nTarget distribution:")
print(y.value_counts())
print("\nPercentages:")
print(y.value_counts(normalize=True) * 100)

# Get numerical columns only
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Remove hash columns - they're just IDs, not useful for prediction
hash_cols = [col for col in numerical_cols if 'hash' in col.lower()]
if hash_cols:
    print(f"\nRemoving hash columns: {len(hash_cols)} found")
    numerical_cols = [col for col in numerical_cols if col not in hash_cols]

print(f"\nUsing {len(numerical_cols)} features for models")

# Prepare feature matrix
X_features = X[numerical_cols].copy()

# Handle any missing data
missing = X_features.isnull().sum().sum()
if missing > 0:
    print(f"Filling {missing} missing values with column means")
    X_features = X_features.fillna(X_features.mean())

print(f"Feature matrix ready: {X_features.shape}")

# Split the data - 70% train, 15% val, 15% test
print("\nSplitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_features, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train) / len(X_features) * 100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val) / len(X_features) * 100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test) / len(X_features) * 100:.1f}%)")

# Scale features
# NOTE: This is important for LR and SVM, not needed for tree models
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check for class imbalance
class_counts = y_train.value_counts()
imbalance_ratio = class_counts.min() / class_counts.max()

print(f"\nChecking class balance...")
print(f"Ratio: {imbalance_ratio:.2f}")

if imbalance_ratio < 0.67:
    print("Classes are imbalanced - using SMOTE to balance")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"Before: {len(X_train_scaled)}")
    print(f"After:  {len(X_train_balanced)}")

    X_train_final = X_train_balanced
    y_train_final = y_train_balanced
else:
    print("Balance looks good, no SMOTE needed")
    X_train_final = X_train_scaled
    y_train_final = y_train

# Train models
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

results = {}

# 1. Logistic Regression - simple baseline
print("\n1. Logistic Regression (baseline)")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_final, y_train_final)

y_pred_lr = lr.predict(X_val_scaled)
y_prob_lr = lr.predict_proba(X_val_scaled)[:, 1]

results['Logistic Regression'] = {
    'model': lr,
    'y_pred': y_pred_lr,
    'y_prob': y_prob_lr,
    'accuracy': accuracy_score(y_val, y_pred_lr),
    'precision': precision_score(y_val, y_pred_lr, average='weighted', zero_division=0),
    'recall': recall_score(y_val, y_pred_lr, average='weighted'),
    'f1': f1_score(y_val, y_pred_lr, average='weighted'),
    'roc_auc': roc_auc_score(y_val, y_prob_lr)
}
print(f"Accuracy: {results['Logistic Regression']['accuracy']:.4f}")

# 2. SVM with RBF kernel
print("\n2. SVM (RBF kernel)")
svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
svm.fit(X_train_final, y_train_final)

y_pred_svm = svm.predict(X_val_scaled)
y_prob_svm = svm.predict_proba(X_val_scaled)[:, 1]

results['SVM'] = {
    'model': svm,
    'y_pred': y_pred_svm,
    'y_prob': y_prob_svm,
    'accuracy': accuracy_score(y_val, y_pred_svm),
    'precision': precision_score(y_val, y_pred_svm, average='weighted', zero_division=0),
    'recall': recall_score(y_val, y_pred_svm, average='weighted'),
    'f1': f1_score(y_val, y_pred_svm, average='weighted'),
    'roc_auc': roc_auc_score(y_val, y_prob_svm)
}
print(f"Accuracy: {results['SVM']['accuracy']:.4f}")

# 3. Random Forest
print("\n3. Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_final, y_train_final)

y_pred_rf = rf.predict(X_val_scaled)
y_prob_rf = rf.predict_proba(X_val_scaled)[:, 1]

results['Random Forest'] = {
    'model': rf,
    'y_pred': y_pred_rf,
    'y_prob': y_prob_rf,
    'accuracy': accuracy_score(y_val, y_pred_rf),
    'precision': precision_score(y_val, y_pred_rf, average='weighted', zero_division=0),
    'recall': recall_score(y_val, y_pred_rf, average='weighted'),
    'f1': f1_score(y_val, y_pred_rf, average='weighted'),
    'roc_auc': roc_auc_score(y_val, y_prob_rf)
}
print(f"Accuracy: {results['Random Forest']['accuracy']:.4f}")

# 4. XGBoost - usually the best performer
print("\n4. XGBoost")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_final, y_train_final)

y_pred_xgb = xgb.predict(X_val_scaled)
y_prob_xgb = xgb.predict_proba(X_val_scaled)[:, 1]

results['XGBoost'] = {
    'model': xgb,
    'y_pred': y_pred_xgb,
    'y_prob': y_prob_xgb,
    'accuracy': accuracy_score(y_val, y_pred_xgb),
    'precision': precision_score(y_val, y_pred_xgb, average='weighted', zero_division=0),
    'recall': recall_score(y_val, y_pred_xgb, average='weighted'),
    'f1': f1_score(y_val, y_pred_xgb, average='weighted'),
    'roc_auc': roc_auc_score(y_val, y_prob_xgb)
}
print(f"Accuracy: {results['XGBoost']['accuracy']:.4f}")

# Compare results
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

comparison = []
for name, metrics in results.items():
    comparison.append({
        'Model': name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'ROC-AUC': metrics['roc_auc']
    })

comparison_df = pd.DataFrame(comparison)
print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(f'{output_dir}/results.csv', index=False)

# Find best model
best_name = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
best_f1 = comparison_df['F1'].max()
print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

# Print detailed reports
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

for name, metrics in results.items():
    print(f"\n{name}:")
    print("-" * 40)
    print(classification_report(y_val, metrics['y_pred'], digits=4))

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, metrics) in enumerate(results.items()):
    cm = confusion_matrix(y_val, metrics['y_pred'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                square=True, linewidths=1, cbar=True)

    axes[idx].set_title(f'{name}\nAcc: {metrics["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300)
plt.close()
print(f"\nSaved confusion matrices")

# ROC curves
plt.figure(figsize=(10, 7))

colors = ['blue', 'red', 'green', 'orange']
for idx, (name, metrics) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_val, metrics['y_prob'], pos_label=y_val.unique()[1])
    plt.plot(fpr, tpr, color=colors[idx], linewidth=2,
             label=f'{name} (AUC={metrics["roc_auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(f'{output_dir}/roc_curves.png', dpi=300)
plt.close()
print("Saved ROC curves")

# Plot metrics comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

for idx, metric in enumerate(metrics_list):
    data = comparison_df.sort_values(metric, ascending=False)
    axes[idx].bar(data['Model'], data[metric], color=colors, alpha=0.7)
    axes[idx].set_title(metric)
    axes[idx].set_ylim([0, 1.1])
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, v in enumerate(data[metric]):
        axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

axes[5].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300)
plt.close()
print("Saved metrics comparison")

# Feature importance for tree models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest
rf_imp = rf.feature_importances_
rf_top = np.argsort(rf_imp)[::-1][:10]

axes[0].barh(range(10), rf_imp[rf_top], color='green', alpha=0.7)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([numerical_cols[i] for i in rf_top])
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest - Top 10 Features')
axes[0].grid(axis='x', alpha=0.3)

# XGBoost
xgb_imp = xgb.feature_importances_
xgb_top = np.argsort(xgb_imp)[::-1][:10]

axes[1].barh(range(10), xgb_imp[xgb_top], color='orange', alpha=0.7)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([numerical_cols[i] for i in xgb_top])
axes[1].invert_yaxis()
axes[1].set_xlabel('Importance')
axes[1].set_title('XGBoost - Top 10 Features')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
plt.close()
print("Saved feature importance")

# Cross-validation
print("\nRunning cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, metrics in results.items():
    scores = cross_val_score(metrics['model'], X_train_final, y_train_final,
                             cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Plot CV results
plt.figure(figsize=(10, 5))
positions = range(len(cv_results))
means = [cv_results[name].mean() for name in results.keys()]
stds = [cv_results[name].std() for name in results.keys()]

plt.bar(positions, means, yerr=stds, color=colors, alpha=0.7, capsize=8)
plt.xticks(positions, results.keys(), rotation=45, ha='right')
plt.ylabel('F1-Score')
plt.title('Cross-Validation Results (5-fold)')
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)

for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.02, f'{m:.3f}', ha='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/cross_validation.png', dpi=300)
plt.close()
print("Saved CV results")

# Test on hold-out set
print("\n" + "=" * 80)
print(f"FINAL TEST - {best_name}")
print("=" * 80)

best_model = results[best_name]['model']
y_test_pred = best_model.predict(X_test_scaled)
y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]

test_results = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred, average='weighted'),
    'Recall': recall_score(y_test, y_test_pred, average='weighted'),
    'F1': f1_score(y_test, y_test_pred, average='weighted'),
    'ROC-AUC': roc_auc_score(y_test, y_test_prob)
}

print("\nTest Set Performance:")
for metric, value in test_results.items():
    print(f"{metric:12s}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

# Test confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            square=True, linewidths=2, cbar=True, annot_kws={'size': 14})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_name} - Test Set\nAcc: {test_results["Accuracy"]:.3f}')
plt.tight_layout()
plt.savefig(f'{output_dir}/test_confusion_matrix.png', dpi=300)
plt.close()
print("\nSaved test results")

# Save model
model_file = f'{output_dir}/best_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(best_model, f)

scaler_file = f'{output_dir}/scaler.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nSaved model: {model_file}")
print(f"Saved scaler: {scaler_file}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nDataset: {len(X_features)} total emails")
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"  Features: {len(numerical_cols)}")

print(f"\nModel Rankings:")
for idx, row in comparison_df.sort_values('F1', ascending=False).iterrows():
    print(f"  {row['Model']:20s}: {row['F1']:.4f}")

print(f"\nBest: {best_name}")
print(f"  Val F1:  {best_f1:.4f}")
print(f"  Test F1: {test_results['F1']:.4f}")
print(f"  Test Acc: {test_results['Accuracy']:.4f}")

if test_results['Accuracy'] > 0.95:
    print("\n✓ Excellent performance - ready for deployment")
elif test_results['Accuracy'] > 0.90:
    print("\n✓ Strong performance - good results")
elif test_results['Accuracy'] > 0.85:
    print("\n✓ Good performance - could be improved")
else:
    print("\n⚠ Performance needs improvement")

print("\n" + "=" * 80)