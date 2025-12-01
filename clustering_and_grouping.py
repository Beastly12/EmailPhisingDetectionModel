import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
OUTPUT_DIR = 'outputs/clustering_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CLUSTERING ANALYSIS FOR PHISHING EMAIL DETECTION")
print("="*80)


# ============================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================
print("\n[STEP 1] Loading and Preprocessing Data")
print("-" * 80)

# Load data
df = pd.read_csv('outputs/Engineered_Features.csv')
print(f"Loaded {len(df)} samples with {len(df.columns)} features")

# Separate labels and features
y_true = df['label'].copy()
X = df.drop(columns=['label'])

print(f"Ground truth distribution: {y_true.value_counts().to_dict()}")

# Select numerical features only (exclude hash columns)
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if 'hash' not in col.lower()]

print(f"Selected {len(numerical_cols)} numerical features")

# Create feature matrix and handle missing values
X_features = X[numerical_cols].fillna(X[numerical_cols].mean())

# Feature Scaling (REQUIRED for distance-based algorithms)
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced dimensions: {X_features.shape[1]} → {X_pca.shape[1]}")
print(f"Total variance preserved: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Visualize PCA variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum_variance)+1), cumsum_variance, 'bo-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA Cumulative Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, min(16, len(pca.explained_variance_ratio_)+1)),
        pca.explained_variance_ratio_[:15], color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Individual Component Variance (Top 15)')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_pca_variance.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/01_pca_variance.png")


# ============================================
# STEP 2: FIND OPTIMAL K
# ============================================
print("\n[STEP 2] Determining Optimal K")
print("-" * 80)

# Test different values of k
k_range = range(2, 11)
silhouette_scores = []
inertias = []

print("Testing k from 2 to 10...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    silhouette_scores.append(silhouette_score(X_pca, labels))
    inertias.append(kmeans.inertia_)

    print(f"k={k}: Silhouette={silhouette_scores[-1]:.3f}, Inertia={inertias[-1]:.2f}")

# Find optimal k (highest silhouette score)
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k = {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")

# Plot k selection
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Silhouette plot
axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2,
                label=f'Optimal k={optimal_k}')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score vs k')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Elbow plot
axes[1].plot(k_range, inertias, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2,
                label=f'Selected k={optimal_k}')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Inertia (WCSS)')
axes[1].set_title('Elbow Method')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_optimal_k.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/02_optimal_k.png")


# ============================================
# STEP 3: RUN CLUSTERING ALGORITHMS
# ============================================
print("\n[STEP 3] Running Clustering Algorithms")
print("-" * 80)

# Dictionary to store results
clustering_results = {}

# 1. K-MEANS
print("\n[3.1] K-Means Clustering")
print(f"Parameters: n_clusters={optimal_k}, random_state=42")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)
clustering_results['KMeans'] = labels_kmeans
print(f"Clusters: {len(np.unique(labels_kmeans))}")
print(f"Cluster sizes: {np.bincount(labels_kmeans)}")

# 2. DBSCAN
print("\n[3.2] DBSCAN Clustering")
print("Finding optimal epsilon using k-distance graph...")

# Find epsilon using k-nearest neighbors
k_neighbors = min(2 * X_pca.shape[1], len(X_pca) - 1)
neighbors = NearestNeighbors(n_neighbors=k_neighbors)
neighbors.fit(X_pca)
distances, _ = neighbors.kneighbors(X_pca)

# Use 90th percentile of k-distances as epsilon
sorted_distances = np.sort(distances[:, k_neighbors-1])
eps = np.percentile(sorted_distances, 90)

print(f"Parameters: eps={eps:.3f}, min_samples={k_neighbors}")
dbscan = DBSCAN(eps=eps, min_samples=k_neighbors)
labels_dbscan = dbscan.fit_predict(X_pca)
clustering_results['DBSCAN'] = labels_dbscan

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)
print(f"Clusters: {n_clusters_dbscan}")
print(f"Noise points: {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)")

# 3. AGGLOMERATIVE CLUSTERING
print("\n[3.3] Agglomerative Hierarchical Clustering")
print(f"Parameters: n_clusters={optimal_k}, linkage='ward'")
agglomerative = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_agg = agglomerative.fit_predict(X_pca)
clustering_results['Agglomerative'] = labels_agg
print(f"Clusters: {len(np.unique(labels_agg))}")
print(f"Cluster sizes: {np.bincount(labels_agg)}")

# Create dendrogram
print("Creating dendrogram...")
linkage_matrix = linkage(X_pca, method='ward')

plt.figure(figsize=(15, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.title('Agglomerative Clustering Dendrogram')
plt.axhline(y=linkage_matrix[-(optimal_k-1), 2], color='r', linestyle='--',
            label=f'Cut at k={optimal_k}')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_dendrogram.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/03_dendrogram.png")

# 4. SPECTRAL CLUSTERING
print("\n[3.4] Spectral Clustering")
if len(X_pca) > 2000:
    print(f"SKIPPED: Dataset too large (n={len(X_pca)} > 2000)")
    print("Reason: Spectral clustering has O(n³) complexity - would take too long")
else:
    print(f"Parameters: n_clusters={optimal_k}, affinity='rbf'")
    spectral = SpectralClustering(n_clusters=optimal_k, affinity='rbf',
                                   random_state=42, n_init=10)
    labels_spectral = spectral.fit_predict(X_pca)
    clustering_results['Spectral'] = labels_spectral
    print(f"Clusters: {len(np.unique(labels_spectral))}")
    print(f"Cluster sizes: {np.bincount(labels_spectral)}")


# ============================================
# STEP 4: EVALUATE CLUSTERING QUALITY
# ============================================
print("\n[STEP 4] Evaluating Clustering Quality")
print("-" * 80)

print("\nMetrics Explained:")
print("  • Silhouette Score [-1, 1]: Higher = better separation (>0.5 is good)")
print("  • Davies-Bouldin [0, ∞): Lower = better (compact & separated)")
print("  • Calinski-Harabasz [0, ∞): Higher = better (dense & separated)")
print("  • Adjusted Rand Index [-1, 1]: Similarity to ground truth (1 = perfect)")
print("  • Normalized Mutual Info [0, 1]: Information shared with ground truth")

evaluation_results = []

for algo_name, labels in clustering_results.items():
    print(f"\n{algo_name}:")

    # Handle DBSCAN noise points
    if algo_name == 'DBSCAN':
        mask = labels >= 0
        X_eval = X_pca[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X_pca
        labels_eval = labels

    # Check if we have enough clusters
    n_clusters = len(set(labels_eval))
    if n_clusters < 2:
        print(f"  Only {n_clusters} cluster - skipping evaluation")
        continue

    # Internal metrics (don't need ground truth)
    silhouette = silhouette_score(X_eval, labels_eval)
    davies_bouldin = davies_bouldin_score(X_eval, labels_eval)
    calinski = calinski_harabasz_score(X_eval, labels_eval)

    # External metrics (compare with ground truth)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print(f"  Silhouette Score:      {silhouette:.4f}")
    print(f"  Davies-Bouldin Index:  {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz:     {calinski:.2f}")
    print(f"  Adjusted Rand Index:   {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")

    evaluation_results.append({
        'Algorithm': algo_name,
        'N_Clusters': n_clusters,
        'N_Noise': list(labels).count(-1),
        'Silhouette': silhouette,
        'Davies_Bouldin': davies_bouldin,
        'Calinski_Harabasz': calinski,
        'ARI': ari,
        'NMI': nmi
    })

# Create evaluation table
eval_df = pd.DataFrame(evaluation_results)

print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(eval_df.to_string(index=False))

# Find best algorithm (highest silhouette)
best_idx = eval_df['Silhouette'].idxmax()
best_algorithm = eval_df.loc[best_idx, 'Algorithm']
print(f"\n★ BEST ALGORITHM: {best_algorithm} (Silhouette: {eval_df.loc[best_idx, 'Silhouette']:.4f})")

# Save metrics
eval_df.to_csv(f'{OUTPUT_DIR}/evaluation_metrics.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/evaluation_metrics.csv")

# Visualize metrics comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')

metrics = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz', 'ARI', 'NMI']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for idx, metric in enumerate(metrics):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    data = eval_df[['Algorithm', metric]].dropna()
    bars = ax.bar(data['Algorithm'], data[metric], color=colors[idx], alpha=0.7)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_metrics_comparison.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/04_metrics_comparison.png")


# ============================================
# STEP 5: CLUSTER PROFILING
# ============================================
print("\n[STEP 5] Cluster Profiling")
print("-" * 80)

print(f"Analyzing {best_algorithm} clusters...")

best_labels = clustering_results[best_algorithm]

# Attach labels to original features
df_profile = X_features.copy()
df_profile['Cluster'] = best_labels

# Calculate mean for each cluster
cluster_profiles = df_profile.groupby('Cluster').mean()

# Calculate Z-scores to find distinguishing features
global_mean = X_features.mean()
global_std = X_features.std().replace(0, 1e-6)  # Avoid division by zero

print("\nCluster Characteristics:")
print("-" * 80)

for cluster in sorted(cluster_profiles.index):
    if cluster == -1:
        print(f"\nCluster: NOISE (outliers)")
    else:
        print(f"\nCluster {cluster}:")

    # Count samples in cluster
    n_samples = (best_labels == cluster).sum()
    pct = n_samples / len(best_labels) * 100
    print(f"  Size: {n_samples} samples ({pct:.1f}%)")

    # Calculate Z-scores
    cluster_means = cluster_profiles.loc[cluster]
    z_scores = (cluster_means - global_mean) / global_std

    # Show top 5 distinguishing features (|Z| > 0.5)
    important = z_scores[abs(z_scores) > 0.5].sort_values(ascending=False).head(5)

    if len(important) > 0:
        print("  Top distinguishing features:")
        for feature, z in important.items():
            direction = "Higher" if z > 0 else "Lower"
            print(f"    • {feature}: {direction} than average (Z={z:+.2f})")
    else:
        print("  No strongly distinguishing features")

# Save cluster profiles
cluster_profiles.to_csv(f'{OUTPUT_DIR}/cluster_profiles.csv')
print(f"\nSaved: {OUTPUT_DIR}/cluster_profiles.csv")

# Show cluster vs ground truth alignment
print("\n" + "-" * 80)
print("Cluster vs Ground Truth Alignment:")
print("-" * 80)

alignment = pd.crosstab(best_labels, y_true, margins=True)
print(alignment)

alignment.to_csv(f'{OUTPUT_DIR}/cluster_alignment.csv')
print(f"Saved: {OUTPUT_DIR}/cluster_alignment.csv")


# ============================================
# STEP 6: VISUALIZATION
# ============================================
print("\n[STEP 6] Creating Visualizations")
print("-" * 80)

# Apply t-SNE for 2D visualization
print("Applying t-SNE for 2D projection (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca)
print("Done!")

# Also create 2D PCA for comparison
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_pca)
print(f"2D PCA variance: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")

# Visualize all algorithms with t-SNE
n_algos = len(clustering_results)
fig, axes = plt.subplots(2, (n_algos+1)//2, figsize=(16, 10))
axes = axes.flatten()

fig.suptitle('Clustering Results (t-SNE Visualization)', fontsize=16, fontweight='bold')

# Plot ground truth first
ax = axes[0]
for label in y_true.unique():
    mask = y_true == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.6, s=30)
ax.set_title('Ground Truth', fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot each clustering algorithm
for idx, (algo_name, labels) in enumerate(clustering_results.items(), start=1):
    ax = axes[idx]

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c='gray', label='Noise', alpha=0.3, s=10, marker='x')
        else:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=[colors[i]], label=f'C{label}', alpha=0.6, s=30)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    title = f'{algo_name}\n({n_clusters} clusters'
    if n_noise > 0:
        title += f', {n_noise} noise)'
    else:
        title += ')'

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(clustering_results) + 1, len(axes)):
    axes[idx].remove()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_tsne_visualization.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/05_tsne_visualization.png")

# Best model visualization (larger)
plt.figure(figsize=(10, 8))
best_labels_vis = clustering_results[best_algorithm]
unique_labels = np.unique(best_labels_vis)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = best_labels_vis == label
    if label == -1:
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c='gray', label='Noise', alpha=0.3, s=20, marker='x')
    else:
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=40)

plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title(f'Best Clustering Result: {best_algorithm}', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_best_model_tsne.png', dpi=300)
plt.close()
print(f"Saved: {OUTPUT_DIR}/06_best_model_tsne.png")


# ============================================
# STEP 7: FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS COMPLETE")
print("="*80)

print(f"\nDataset: {len(X_features)} samples, {X_features.shape[1]} features")
print(f"PCA reduced to: {X_pca.shape[1]} dimensions")
print(f"Optimal k: {optimal_k}")
print(f"Best algorithm: {best_algorithm}")

print("\nAlgorithm Rankings (by Silhouette Score):")
ranked = eval_df.sort_values('Silhouette', ascending=False)
for i, row in ranked.iterrows():
    print(f"  {row['Algorithm']:15s}: {row['Silhouette']:.4f}")

print("\nInterpretation:")
print("  • Higher Silhouette Score = Better cluster separation")
print(f"  • {best_algorithm} achieved the best clustering quality")
print("  • Check cluster profiles to understand what each cluster represents")

print("\nNext Steps:")
print("  1. Review visualizations in outputs/clustering_analysis/")
print("  2. Analyze cluster_profiles.csv to interpret clusters")
print("  3. Use cluster labels as features for ML models")

print("\n" + "="*80)
print("All outputs saved to:", OUTPUT_DIR)
print("="*80)
