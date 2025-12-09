"""
Galaxy Morphology Analysis - Step 6: Unsupervised Learning
Clustering + Anomaly Detection to find rare/odd galaxies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Create output directories
os.makedirs("outputs/unsupervised", exist_ok=True)
os.makedirs("outputs/unsupervised/clustering", exist_ok=True)
os.makedirs("outputs/unsupervised/anomalies", exist_ok=True)

print("=" * 80)
print("UNSUPERVISED LEARNING: CLUSTERING & ANOMALY DETECTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA AND PREPARE FEATURES
# ============================================================================

print("\n[1/6] Loading dataset...")
ds = load_from_disk("gz2_prepared")
df = ds.to_pandas()
print(f"✓ Loaded {len(df):,} galaxies")

# Select feature columns
fraction_cols = [c for c in df.columns if c.endswith("_fraction")]
label_cols = [c for c in df.columns if c.endswith("_label")]

print(f"\n   Features: {len(fraction_cols)} fraction columns")
print(f"   Labels: {len(label_cols)} label columns")

# Prepare feature matrix
X = df[fraction_cols].fillna(0).values
print(f"\n   Feature matrix shape: {X.shape}")

# Create true galaxy type labels for evaluation
def create_target(row):
    if row['spiral_label'] == 1:
        return 2  # Spiral
    elif row['disk_label'] == 1:
        return 1  # Disk
    else:
        return 0  # Smooth

df['galaxy_type'] = df.apply(create_target, axis=1)
y_true = df['galaxy_type'].values

class_names = {0: 'Smooth', 1: 'Disk', 2: 'Spiral'}
print("\n   True class distribution:")
for cls in [0, 1, 2]:
    count = (y_true == cls).sum()
    pct = (count / len(y_true)) * 100
    print(f"     {class_names[cls]}: {count:,} ({pct:.1f}%)")

# Standardize features
print("\n[2/6] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 2: DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================================

print("\n[3/6] Applying dimensionality reduction...")

# PCA
# PCA - adjust n_components to not exceed number of features
n_features = X_scaled.shape[1]
n_pca_components = min(40, n_features)  # Use at most 40 or number of features

print(f"   Running PCA (using {n_pca_components} components)...")
pca = PCA(n_components=n_pca_components)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

# Adjust indexing based on actual components
if n_pca_components >= 10:
    print(f"   ✓ PCA: {cumsum_var[9]:.2%} variance explained by 10 components")
if n_pca_components >= 20:
    print(f"   ✓ PCA: {cumsum_var[19]:.2%} variance explained by 20 components")
print(f"   ✓ PCA: {cumsum_var[-1]:.2%} variance explained by all {n_pca_components} components")

# t-SNE for visualization
print("   Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_umap = tsne.fit_transform(X_scaled)  # Keep variable name X_umap so rest works
print("   ✓ t-SNE completed")

# Save embeddings
embeddings_df = pd.DataFrame({
    'tsne_1': X_umap[:, 0],
    'tsne_2': X_umap[:, 1],
    'galaxy_type': y_true,
    'odd_label': df['odd_label'].values
})

# ============================================================================
# STEP 3: CLUSTERING ANALYSIS
# ============================================================================

print("\n[4/6] Performing clustering analysis...")

clustering_results = {}

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

print("\n   [K-Means] Testing different k values...")
k_values = [3, 5, 7, 10]
kmeans_metrics = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca[:, :10])  # Use top 10 PCA components
    
    # Compute metrics
    silhouette = silhouette_score(X_pca[:, :10], kmeans_labels)
    db_score = davies_bouldin_score(X_pca[:, :10], kmeans_labels)
    ari = adjusted_rand_score(y_true, kmeans_labels)
    
    kmeans_metrics.append({
        'k': k,
        'silhouette': silhouette,
        'davies_bouldin': db_score,
        'ARI': ari
    })
    
    print(f"      k={k}: Silhouette={silhouette:.3f}, DB={db_score:.3f}, ARI={ari:.3f}")
    
    if k == 3:  # Save k=3 for comparison with true labels
        clustering_results['kmeans_k3'] = kmeans_labels

kmeans_df = pd.DataFrame(kmeans_metrics)
kmeans_df.to_csv("outputs/unsupervised/clustering/kmeans_metrics.csv", index=False)

# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================

print("\n   [DBSCAN] Finding density-based clusters...")
# Use UMAP embeddings for DBSCAN (works better in low-D)
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_umap)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"      Clusters found: {n_clusters}")
print(f"      Noise points: {n_noise:,} ({n_noise/len(dbscan_labels)*100:.1f}%)")

if n_clusters > 1:
    # Silhouette only if we have clusters
    mask = dbscan_labels != -1
    if mask.sum() > 0:
        silhouette = silhouette_score(X_umap[mask], dbscan_labels[mask])
        print(f"      Silhouette score: {silhouette:.3f}")

clustering_results['dbscan'] = dbscan_labels

# ============================================================================
# STEP 4: ANOMALY DETECTION
# ============================================================================

print("\n[5/6] Running anomaly detection methods...")

anomaly_results = {}

# ============================================================================
# ISOLATION FOREST
# ============================================================================

print("\n   [Isolation Forest]")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.score_samples(X_scaled)

n_anomalies = (iso_predictions == -1).sum()
print(f"      Anomalies detected: {n_anomalies:,} ({n_anomalies/len(X)*100:.1f}%)")

anomaly_results['isolation_forest'] = {
    'predictions': iso_predictions,
    'scores': iso_scores
}

# ============================================================================
# LOCAL OUTLIER FACTOR
# ============================================================================

print("\n   [Local Outlier Factor]")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_predictions = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

n_anomalies = (lof_predictions == -1).sum()
print(f"      Anomalies detected: {n_anomalies:,} ({n_anomalies/len(X)*100:.1f}%)")

anomaly_results['lof'] = {
    'predictions': lof_predictions,
    'scores': lof_scores
}

# ============================================================================
# PCA RECONSTRUCTION ERROR
# ============================================================================

print("\n   [PCA Reconstruction Error]")
# Use fewer components to detect anomalies (high reconstruction error = anomaly)
pca_small = PCA(n_components=10)
X_pca_small = pca_small.fit_transform(X_scaled)
X_reconstructed = pca_small.inverse_transform(X_pca_small)

# Compute reconstruction error
reconstruction_errors = np.sqrt(np.sum((X_scaled - X_reconstructed)**2, axis=1))

# Define anomalies as top 5% reconstruction errors
threshold = np.percentile(reconstruction_errors, 95)
pca_predictions = np.where(reconstruction_errors > threshold, -1, 1)

n_anomalies = (pca_predictions == -1).sum()
print(f"      Anomalies detected: {n_anomalies:,} ({n_anomalies/len(X)*100:.1f}%)")
print(f"      Reconstruction error threshold: {threshold:.4f}")

anomaly_results['pca'] = {
    'predictions': pca_predictions,
    'scores': reconstruction_errors
}

# ============================================================================
# COMPARE WITH "ODD" GALAXIES
# ============================================================================

print("\n   [Validation] Comparing with known 'odd' galaxies...")
odd_mask = df['odd_label'].values == 1
n_odd = odd_mask.sum()
print(f"      Known odd galaxies: {n_odd:,} ({n_odd/len(df)*100:.1f}%)")

for method_name, result in anomaly_results.items():
    detected = result['predictions'] == -1
    overlap = (detected & odd_mask).sum()
    precision = overlap / detected.sum() if detected.sum() > 0 else 0
    recall = overlap / n_odd if n_odd > 0 else 0
    print(f"\n      {method_name.upper()}:")
    print(f"         Detected odd galaxies: {overlap:,}")
    print(f"         Precision: {precision:.2%}")
    print(f"         Recall: {recall:.2%}")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n[6/6] Saving results...")

# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'galaxy_id': range(len(df)),
    'galaxy_type': y_true,
    'odd_label': df['odd_label'].values,
    'smooth_fraction': df['smooth_fraction'].values,
    'disk_fraction': df['disk_fraction'].values,
    'spiral_fraction': df['spiral_fraction'].values,
    'tsne_1': X_umap[:, 0],
    'tsne_2': X_umap[:, 1],
    'kmeans_k3': clustering_results['kmeans_k3'],
    'dbscan_cluster': clustering_results['dbscan'],
    'iso_forest_anomaly': (anomaly_results['isolation_forest']['predictions'] == -1).astype(int),
    'iso_forest_score': anomaly_results['isolation_forest']['scores'],
    'lof_anomaly': (anomaly_results['lof']['predictions'] == -1).astype(int),
    'lof_score': anomaly_results['lof']['scores'],
    'pca_anomaly': (anomaly_results['pca']['predictions'] == -1).astype(int),
    'pca_reconstruction_error': anomaly_results['pca']['scores']
})

results_df.to_csv("outputs/unsupervised/unsupervised_results.csv", index=False)
print("✓ Saved unsupervised_results.csv")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

# Set style
sns.set_style("whitegrid")

# ============================================================================
# 1. UMAP VISUALIZATION WITH TRUE LABELS
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# True galaxy types
for cls in [0, 1, 2]:
    mask = y_true == cls
    axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                   alpha=0.3, s=5, label=class_names[cls])
axes[0].set_xlabel('TSNE 1', fontsize=12)
axes[0].set_ylabel('TSNE 2', fontsize=12)
axes[0].set_title('True Galaxy Types', fontsize=14, fontweight='bold')
axes[0].legend()

# K-Means clusters (k=3)
scatter = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], 
                         c=clustering_results['kmeans_k3'], 
                         cmap='viridis', alpha=0.3, s=5)
axes[1].set_xlabel('TSNE 1', fontsize=12)
axes[1].set_ylabel('TSNE 2', fontsize=12)
axes[1].set_title('K-Means Clustering (k=3)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[1], label='Cluster')

# Odd galaxies
odd_mask = df['odd_label'].values == 1
axes[2].scatter(X_umap[~odd_mask, 0], X_umap[~odd_mask, 1], 
               c='lightgray', alpha=0.2, s=5, label='Normal')
axes[2].scatter(X_umap[odd_mask, 0], X_umap[odd_mask, 1], 
               c='red', alpha=0.5, s=10, label='Odd')
axes[2].set_xlabel('TSNE 1', fontsize=12)
axes[2].set_ylabel('TSNE 2', fontsize=12)
axes[2].set_title('Known "Odd" Galaxies', fontsize=14, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/unsupervised/clustering/umap_visualization.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved tsne_visualization.png")
plt.close()

# ============================================================================
# 2. ANOMALY DETECTION VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Isolation Forest
iso_anomalies = anomaly_results['isolation_forest']['predictions'] == -1
axes[0, 0].scatter(X_umap[~iso_anomalies, 0], X_umap[~iso_anomalies, 1],
                  c='lightblue', alpha=0.2, s=5, label='Normal')
axes[0, 0].scatter(X_umap[iso_anomalies, 0], X_umap[iso_anomalies, 1],
                  c='red', alpha=0.7, s=15, label='Anomaly')
axes[0, 0].set_xlabel('TSNE 1', fontsize=11)
axes[0, 0].set_ylabel('TSNE 2', fontsize=11)
axes[0, 0].set_title('Isolation Forest Anomalies', fontsize=13, fontweight='bold')
axes[0, 0].legend()

# LOF
lof_anomalies = anomaly_results['lof']['predictions'] == -1
axes[0, 1].scatter(X_umap[~lof_anomalies, 0], X_umap[~lof_anomalies, 1],
                  c='lightblue', alpha=0.2, s=5, label='Normal')
axes[0, 1].scatter(X_umap[lof_anomalies, 0], X_umap[lof_anomalies, 1],
                  c='orange', alpha=0.7, s=15, label='Anomaly')
axes[0, 1].set_xlabel('TSNE 1', fontsize=11)
axes[0, 1].set_ylabel('TSNE 2', fontsize=11)
axes[0, 1].set_title('Local Outlier Factor Anomalies', fontsize=13, fontweight='bold')
axes[0, 1].legend()

# PCA Reconstruction Error
pca_anomalies = anomaly_results['pca']['predictions'] == -1
axes[1, 0].scatter(X_umap[~pca_anomalies, 0], X_umap[~pca_anomalies, 1],
                  c='lightblue', alpha=0.2, s=5, label='Normal')
axes[1, 0].scatter(X_umap[pca_anomalies, 0], X_umap[pca_anomalies, 1],
                  c='purple', alpha=0.7, s=15, label='Anomaly')
axes[1, 0].set_xlabel('TSNE 1', fontsize=11)
axes[1, 0].set_ylabel('TSNE 2', fontsize=11)
axes[1, 0].set_title('PCA Reconstruction Error Anomalies', fontsize=13, fontweight='bold')
axes[1, 0].legend()

# Agreement between methods
iso_set = set(np.where(iso_anomalies)[0])
lof_set = set(np.where(lof_anomalies)[0])
pca_set = set(np.where(pca_anomalies)[0])
all_three = iso_set & lof_set & pca_set
any_one = iso_set | lof_set | pca_set

consensus_mask = np.zeros(len(X), dtype=bool)
consensus_mask[list(all_three)] = True

axes[1, 1].scatter(X_umap[~consensus_mask, 0], X_umap[~consensus_mask, 1],
                  c='lightblue', alpha=0.2, s=5, label='Normal')
axes[1, 1].scatter(X_umap[consensus_mask, 0], X_umap[consensus_mask, 1],
                  c='darkred', alpha=0.9, s=20, marker='*', 
                  label=f'Consensus ({len(all_three)} galaxies)')
axes[1, 1].set_xlabel('TSNE 1', fontsize=11)
axes[1, 1].set_ylabel('TSNE 2', fontsize=11)
axes[1, 1].set_title('Consensus Anomalies (All 3 Methods)', fontsize=13, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('outputs/unsupervised/anomalies/anomaly_detection.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved anomaly_detection.png")
plt.close()

# ============================================================================
# 3. PCA VARIANCE PLOT
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
n_components_plot = len(explained_var)
axes[0].plot(range(1, n_components_plot + 1), explained_var, 'bo-', linewidth=2)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, n_components_plot + 1), cumsum_var, 'ro-', linewidth=2)
axes[1].axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/unsupervised/clustering/pca_variance.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved pca_variance.png")
plt.close()

# ============================================================================
# 4. K-MEANS METRICS
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Silhouette score
axes[0].plot(kmeans_df['k'], kmeans_df['silhouette'], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].set_title('K-Means: Silhouette Score', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Davies-Bouldin Index
axes[1].plot(kmeans_df['k'], kmeans_df['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[1].set_title('K-Means: Davies-Bouldin Index (lower is better)', 
                 fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

# Adjusted Rand Index
axes[2].plot(kmeans_df['k'], kmeans_df['ARI'], 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[2].set_ylabel('Adjusted Rand Index', fontsize=12)
axes[2].set_title('K-Means: ARI vs True Labels', fontsize=14, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('unsupervised/clustering/kmeans_evaluation.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved kmeans_evaluation.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("UNSUPERVISED LEARNING SUMMARY")
print("=" * 80)

print("\n📊 CLUSTERING RESULTS:")
print(f"   K-Means (k=3): Best silhouette = {kmeans_df.loc[0, 'silhouette']:.3f}")
print(f"   DBSCAN: {n_clusters} clusters, {n_noise:,} noise points")

print("\n🔍 ANOMALY DETECTION RESULTS:")
print(f"   Isolation Forest: {(iso_predictions == -1).sum():,} anomalies")
print(f"   LOF: {(lof_predictions == -1).sum():,} anomalies")
print(f"   PCA Reconstruction: {(pca_predictions == -1).sum():,} anomalies")
print(f"   Consensus (all 3): {len(all_three):,} anomalies")

print("\n✨ TOP 10 MOST ANOMALOUS GALAXIES (by consensus):")
if len(all_three) > 0:
    consensus_indices = list(all_three)[:10]
    for i, idx in enumerate(consensus_indices, 1):
        print(f"   {i}. Galaxy {idx}: "
              f"Smooth={df.iloc[idx]['smooth_fraction']:.2f}, "
              f"Disk={df.iloc[idx]['disk_fraction']:.2f}, "
              f"Spiral={df.iloc[idx]['spiral_fraction']:.2f}, "
              f"Odd={df.iloc[idx]['odd_fraction']:.2f}")

print("\n" + "=" * 80)
print("FILES SAVED:")
print("=" * 80)
print("   outputs/unsupervised/unsupervised_results.csv")
print("   outputs/unsupervised/clustering/umap_visualization.png")
print("   outputs/unsupervised/clustering/pca_variance.png")
print("   outputs/unsupervised/clustering/kmeans_evaluation.png")
print("   outputs/unsupervised/anomalies/anomaly_detection.png")
print("=" * 80)

print("\n✅ Unsupervised learning analysis complete!")