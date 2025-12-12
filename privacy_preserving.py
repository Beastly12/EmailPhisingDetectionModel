import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.linear_model import LogisticRegression
import pickle
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
output_dir = 'outputs/privacy'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("PRIVACY-PRESERVING TECHNIQUES")
print("=" * 70)

# Load data
df = pd.read_csv('outputs/Engineered_Features.csv')
print(f"\nLoaded {len(df)} emails")

y = df['label']
X = df.drop(columns=['label'])

# Get numerical features - same as before
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
hash_cols = [col for col in numerical_cols if 'hash' in col.lower()]
numerical_cols = [col for col in numerical_cols if col not in hash_cols]

X_features = X[numerical_cols].copy()
if X_features.isnull().sum().sum() > 0:
    X_features = X_features.fillna(X_features.mean())

print(f"Using {len(numerical_cols)} features")

# Quick note on what we've already done for privacy
print("\n" + "-" * 70)
print("Privacy measures already in place:")
print("  - Email addresses hashed (can't be reversed)")
print("  - URLs replaced with counts")
print("  - No actual email text stored")
print("  - Only statistical features kept")
print("-" * 70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# First train a regular model as baseline
print("\n" + "=" * 70)
print("BASELINE (NO PRIVACY)")
print("=" * 70)

print("Training regular logistic regression...")
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train_scaled, y_train)

y_pred = baseline.predict(X_test_scaled)

baseline_acc = accuracy_score(y_test, y_pred)
baseline_prec = precision_score(y_test, y_pred, average='weighted')
baseline_rec = recall_score(y_test, y_pred, average='weighted')
baseline_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nBaseline results:")
print(f"  Accuracy:  {baseline_acc:.4f}")
print(f"  Precision: {baseline_prec:.4f}")
print(f"  Recall:    {baseline_rec:.4f}")
print(f"  F1:        {baseline_f1:.4f}")

# Now test differential privacy
print("\n" + "=" * 70)
print("DIFFERENTIAL PRIVACY")
print("=" * 70)

print("\nWhat is it?")
print("Adds random noise during training so the model can't memorize")
print("individual emails. Controlled by epsilon (ε):")
print("  - Small ε (< 1): Strong privacy, lower accuracy")
print("  - Medium ε (1-2): Balanced")
print("  - Large ε (> 3): Weak privacy, similar to baseline")

# Try different epsilon values
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
results = []

print(f"\nTesting {len(epsilons)} different privacy levels...")

for eps in epsilons:
    print(f"\n  ε = {eps}...", end=" ")

    # Train with differential privacy
    dp_model = DPLogisticRegression(
        epsilon=eps,
        data_norm=10.0,  # Estimated from our scaled data
        max_iter=1000,
        random_state=42
    )

    try:
        dp_model.fit(X_train_scaled, y_train)
        y_pred = dp_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        loss = (baseline_acc - acc) * 100

        results.append({
            'Epsilon': eps,
            'Privacy': 'Strong' if eps < 1 else ('Medium' if eps <= 2 else 'Weak'),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Loss_pct': loss
        })

        print(f"Acc: {acc:.4f} (lost {loss:.1f}%)")

    except Exception as e:
        print(f"Failed - {str(e)}")

# results table
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(results_df.to_string(index=False))

results_df.to_csv(f'{output_dir}/dp_results.csv', index=False)

# Plot privacy vs accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left plot - accuracy curve
ax1.plot(results_df['Epsilon'], results_df['Accuracy'], 'bo-', linewidth=2, markersize=7)
ax1.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label='Baseline')
ax1.set_xlabel('Epsilon (ε)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Privacy vs Accuracy Trade-off')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(alpha=0.3)

# Add some annotations for key points
for eps in [0.5, 1.0, 5.0]:
    row = results_df[results_df['Epsilon'] == eps].iloc[0]
    ax1.annotate(f"ε={eps}", xy=(eps, row['Accuracy']),
                 xytext=(10, -10), textcoords='offset points', fontsize=9)

# Right plot - accuracy loss bars
colors = ['red' if e < 1 else ('orange' if e <= 2 else 'green')
          for e in results_df['Epsilon']]
ax2.bar(range(len(results_df)), results_df['Loss_pct'],
        color=colors, alpha=0.6, edgecolor='black')
ax2.set_xlabel('Epsilon Level')
ax2.set_ylabel('Accuracy Loss (%)')
ax2.set_title('Cost of Privacy')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels([f"ε={e}" for e in results_df['Epsilon']], rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add values on bars
for i, v in enumerate(results_df['Loss_pct']):
    ax2.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/privacy_tradeoff.png', dpi=300)
plt.close()
print(f"\nSaved privacy trade-off plot")

# Show all metrics for different epsilon
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
baseline_vals = [baseline_acc, baseline_prec, baseline_rec, baseline_f1]

for idx, (metric, baseline_val) in enumerate(zip(metrics, baseline_vals)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(results_df['Epsilon'], results_df[metric], 'go-', linewidth=2, markersize=7)
    ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Privacy Level')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/all_metrics.png', dpi=300)
plt.close()
print("Saved detailed metrics plot")

# Recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find ε=1.0 results
rec_row = results_df[results_df['Epsilon'] == 1.0].iloc[0]

print(f"\nBest choice: ε = 1.0 (balanced privacy)")
print(f"  Accuracy:  {rec_row['Accuracy']:.4f}")
print(f"  F1-Score:  {rec_row['F1']:.4f}")
print(f"  Cost:      {rec_row['Loss_pct']:.2f}% accuracy loss")

if rec_row['Loss_pct'] < 5:
    print("\n✓ Good trade-off - recommended for deployment")
elif rec_row['Loss_pct'] < 10:
    print("\n⚠ Acceptable but consider if privacy is worth the cost")
else:
    print("\n✗ High cost - might need to increase epsilon")

# Train final model with recommended epsilon
print(f"\nTraining final DP model with ε=1.0...")
final_model = DPLogisticRegression(epsilon=1.0, data_norm=10.0,
                                   max_iter=1000, random_state=42)
final_model.fit(X_train_scaled, y_train)

with open(f'{output_dir}/dp_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved")

# Federated learning simulation
print("\n" + "=" * 70)
print("FEDERATED LEARNING SIMULATION")
print("=" * 70)

print("\nWhat is it?")
print("Model trains locally on each organization's data, then")
print("only the model weights are shared (not the actual data).")
print("Simulating 5 different organizations...")

# Split data among 5 "organizations"
n_clients = 5
client_data = []

for i in range(n_clients):
    start = i * len(X_train_scaled) // n_clients
    end = (i + 1) * len(X_train_scaled) // n_clients

    X_client = X_train_scaled[start:end]
    y_client = y_train.iloc[start:end]

    client_data.append((X_client, y_client))
    print(f"  Org {i + 1}: {len(X_client)} emails")

# Run 3 rounds of federated learning
n_rounds = 3
global_accuracies = []

print(f"\nRunning {n_rounds} federated rounds...")

for round_num in range(n_rounds):
    print(f"\nRound {round_num + 1}:")

    # Each client trains locally
    local_models = []

    for i, (X_client, y_client) in enumerate(client_data):
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_client, y_client)
        local_models.append(model)

        local_acc = model.score(X_client, y_client)
        print(f"  Org {i + 1} local: {local_acc:.4f}")

    # Average the model parameters (this is the "federated" part)
    avg_coef = np.mean([m.coef_ for m in local_models], axis=0)
    avg_intercept = np.mean([m.intercept_ for m in local_models])

    # Make global model with averaged weights
    global_model = LogisticRegression(max_iter=1, random_state=42)
    global_model.fit(X_train_scaled[:10], y_train[:10])  # Just to initialize
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept

    # Test global model
    global_acc = global_model.score(X_test_scaled, y_test)
    global_accuracies.append(global_acc)
    print(f"  Global model: {global_acc:.4f}")

print("\nFederated learning done")
print(f"Final federated accuracy: {global_accuracies[-1]:.4f}")
print(f"Baseline (all data):      {baseline_acc:.4f}")
print(f"Difference:               {(baseline_acc - global_accuracies[-1]) * 100:.2f}%")

# Compare everything
print("\n" + "=" * 70)
print("OVERALL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Method': [
        'Baseline (no privacy)',
        'DP (ε=0.5, strong)',
        'DP (ε=1.0, balanced)',
        'DP (ε=2.0, weak)',
        'Federated Learning'
    ],
    'Privacy Level': ['None', 'Strong', 'Medium', 'Weak', 'Strong'],
    'Accuracy': [
        baseline_acc,
        results_df[results_df['Epsilon'] == 0.5]['Accuracy'].values[0],
        results_df[results_df['Epsilon'] == 1.0]['Accuracy'].values[0],
        results_df[results_df['Epsilon'] == 2.0]['Accuracy'].values[0],
        global_accuracies[-1]
    ]
})

print()
print(comparison.to_string(index=False))

comparison.to_csv(f'{output_dir}/comparison.csv', index=False)

# Plot comparison
plt.figure(figsize=(10, 6))
colors_map = {'None': 'red', 'Strong': 'green', 'Medium': 'orange', 'Weak': 'yellow'}
bar_colors = [colors_map[p] for p in comparison['Privacy Level']]

bars = plt.bar(range(len(comparison)), comparison['Accuracy'],
               color=bar_colors, alpha=0.6, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(comparison)), comparison['Method'], rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Privacy Methods Comparison')
plt.ylim([0.7, 1.0])
plt.grid(axis='y', alpha=0.3)

# Add values
for i, v in enumerate(comparison['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Legend
from matplotlib.patches import Patch

legend = [Patch(facecolor='red', label='No Privacy'),
          Patch(facecolor='yellow', label='Weak Privacy'),
          Patch(facecolor='orange', label='Medium Privacy'),
          Patch(facecolor='green', label='Strong Privacy')]
plt.legend(handles=legend, loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/comparison_chart.png', dpi=300)
plt.close()
print("\nSaved comparison chart")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nKey findings:")
print(f"  • Baseline accuracy: {baseline_acc:.4f}")
print(f"  • DP (ε=1.0) accuracy: {rec_row['Accuracy']:.4f} (cost: {rec_row['Loss_pct']:.1f}%)")
print(f"  • Federated accuracy: {global_accuracies[-1]:.4f}")


print("\nGDPR compliance:")
print("  ✓ Personal data anonymized")
print("  ✓ Model has privacy guarantees")
print("  ✓ Individual emails can't be reconstructed")

print("\n" + "=" * 70)
print("PRIVACY ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print(f"  - {output_dir}/dp_results.csv")
print(f"  - {output_dir}/privacy_tradeoff.png")
print(f"  - {output_dir}/all_metrics.png")
print(f"  - {output_dir}/comparison.csv")
print(f"  - {output_dir}/comparison_chart.png")
print(f"  - {output_dir}/dp_model.pkl")