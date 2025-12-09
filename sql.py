"""
Galaxy Morphology Analysis - Step 7: SQL Database Integration
Store and query results using SQLite
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import os

# Create output directory
os.makedirs("outputs/sql", exist_ok=True)

print("=" * 80)
print("SQL DATABASE INTEGRATION FOR GALAXY MORPHOLOGY ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: CREATE DATABASE AND SCHEMA
# ============================================================================

print("\n[1/6] Creating SQLite database...")

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect('outputs/sql/galaxy_morphology.db')
cursor = conn.cursor()

print("✓ Connected to database: galaxy_morphology.db")

# Drop existing tables if they exist
cursor.execute("DROP TABLE IF EXISTS galaxies")
cursor.execute("DROP TABLE IF EXISTS ml_predictions")
cursor.execute("DROP TABLE IF EXISTS clustering_results")
cursor.execute("DROP TABLE IF EXISTS anomaly_scores")
cursor.execute("DROP TABLE IF EXISTS model_performance")

print("\n[2/6] Creating database schema...")

# Table 1: Galaxy metadata and features
cursor.execute("""
CREATE TABLE galaxies (
    galaxy_id INTEGER PRIMARY KEY,
    smooth_fraction REAL,
    disk_fraction REAL,
    spiral_fraction REAL,
    edge_on_fraction REAL,
    odd_fraction REAL,
    smooth_label INTEGER,
    disk_label INTEGER,
    spiral_label INTEGER,
    odd_label INTEGER,
    galaxy_type INTEGER,
    galaxy_type_name TEXT
)
""")

# Table 2: ML model predictions
cursor.execute("""
CREATE TABLE ml_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    galaxy_id INTEGER,
    model_name TEXT,
    predicted_class INTEGER,
    predicted_class_name TEXT,
    confidence REAL,
    correct INTEGER,
    FOREIGN KEY (galaxy_id) REFERENCES galaxies(galaxy_id)
)
""")

# Table 3: Clustering results
cursor.execute("""
CREATE TABLE clustering_results (
    galaxy_id INTEGER PRIMARY KEY,
    kmeans_k3_cluster INTEGER,
    dbscan_cluster INTEGER,
    tsne_1 REAL,
    tsne_2 REAL,
    FOREIGN KEY (galaxy_id) REFERENCES galaxies(galaxy_id)
)
""")

# Table 4: Anomaly detection scores
cursor.execute("""
CREATE TABLE anomaly_scores (
    galaxy_id INTEGER PRIMARY KEY,
    isolation_forest_anomaly INTEGER,
    isolation_forest_score REAL,
    lof_anomaly INTEGER,
    lof_score REAL,
    pca_anomaly INTEGER,
    pca_reconstruction_error REAL,
    consensus_anomaly INTEGER,
    FOREIGN KEY (galaxy_id) REFERENCES galaxies(galaxy_id)
)
""")

# Table 5: Model performance metrics
cursor.execute("""
CREATE TABLE model_performance (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    accuracy REAL,
    f1_macro REAL,
    f1_weighted REAL,
    precision_macro REAL,
    recall_macro REAL,
    auroc REAL,
    training_date TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
print("✓ Created 5 tables:")
print("   - galaxies")
print("   - ml_predictions")
print("   - clustering_results")
print("   - anomaly_scores")
print("   - model_performance")

# ============================================================================
# STEP 2: LOAD AND INSERT DATA
# ============================================================================

print("\n[3/6] Loading data from previous analyses...")

# ============================================================================
# 🔴 UPDATE THIS PATH - Load original dataset
# ============================================================================
ds = load_from_disk("gz2_prepared")  # ← CHANGE THIS PATH
df = ds.to_pandas()

# Create galaxy_type
def create_target(row):
    if row['spiral_label'] == 1:
        return 2
    elif row['disk_label'] == 1:
        return 1
    else:
        return 0

df['galaxy_type'] = df.apply(create_target, axis=1)
df['galaxy_type_name'] = df['galaxy_type'].map({0: 'Smooth', 1: 'Disk', 2: 'Spiral'})

# Insert galaxy data
print("\n   Inserting galaxy metadata...")
galaxy_data = df[['smooth_fraction', 'disk_fraction', 'spiral_fraction', 
                  'edge_on_fraction', 'odd_fraction', 
                  'smooth_label', 'disk_label', 'spiral_label', 'odd_label',
                  'galaxy_type', 'galaxy_type_name']].copy()
galaxy_data['galaxy_id'] = range(len(galaxy_data))

galaxy_data.to_sql('galaxies', conn, if_exists='append', index=False)
print(f"   ✓ Inserted {len(galaxy_data):,} galaxy records")

# ============================================================================
# 🔴 UPDATE THIS PATH - Load clustering results
# ============================================================================
print("\n   Inserting clustering results...")
clustering_df = pd.read_csv('outputs/unsupervised/unsupervised_results.csv')  # ← CHANGE THIS PATH IF NEEDED
clustering_data = clustering_df[['galaxy_id', 'kmeans_k3', 'dbscan_cluster', 'tsne_1', 'tsne_2']].copy()
clustering_data.columns = ['galaxy_id', 'kmeans_k3_cluster', 'dbscan_cluster', 'tsne_1', 'tsne_2']
clustering_data.to_sql('clustering_results', conn, if_exists='append', index=False)
print(f"   ✓ Inserted {len(clustering_data):,} clustering records")

# Load and insert anomaly scores
print("\n   Inserting anomaly detection scores...")
anomaly_data = clustering_df[['galaxy_id', 'iso_forest_anomaly', 'iso_forest_score',
                               'lof_anomaly', 'lof_score', 'pca_anomaly', 
                               'pca_reconstruction_error']].copy()

# Add consensus anomaly
anomaly_data['consensus_anomaly'] = (
    (anomaly_data['iso_forest_anomaly'] == 1) & 
    (anomaly_data['lof_anomaly'] == 1) & 
    (anomaly_data['pca_anomaly'] == 1)
).astype(int)

anomaly_data.columns = ['galaxy_id', 'isolation_forest_anomaly', 'isolation_forest_score',
                        'lof_anomaly', 'lof_score', 'pca_anomaly', 
                        'pca_reconstruction_error', 'consensus_anomaly']
anomaly_data.to_sql('anomaly_scores', conn, if_exists='append', index=False)
print(f"   ✓ Inserted {len(anomaly_data):,} anomaly score records")

# ============================================================================
# 🔴 LOAD ML MODEL PERFORMANCE FROM YOUR CSV FILES
# ============================================================================
print("\n   Inserting model performance metrics...")

# Load your comprehensive_metrics.csv
ml_metrics = pd.read_csv('outputs/results/comprehensive_metrics.csv')  # ← CHANGE THIS PATH

# Load your CNN results
cnn_metrics = pd.read_csv('outputs/cnn/cnn_results.csv')  # ← CHANGE THIS PATH

# Prepare ML models data
for _, row in ml_metrics.iterrows():
    cursor.execute("""
        INSERT INTO model_performance 
        (model_name, accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auroc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        row['Model'], 
        row['Accuracy'], 
        row['F1 (Macro)'], 
        row['F1 (Weighted)'],
        row['Precision (Macro)'],
        row['Recall (Macro)'],
        row['AUROC (OvR)']
    ))

# Prepare CNN data (note: your CNN CSV doesn't have Precision and Recall)
for _, row in cnn_metrics.iterrows():
    cursor.execute("""
        INSERT INTO model_performance 
        (model_name, accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auroc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        row['Model'],
        row['Accuracy'],
        row['F1 (Macro)'],
        row['F1 (Weighted)'],
        None,  # Precision not available in CNN CSV
        None,  # Recall not available in CNN CSV
        row['AUROC (OvR)']
    ))

conn.commit()
print(f"   ✓ Inserted {len(ml_metrics) + len(cnn_metrics)} model performance records")

# ============================================================================
# STEP 3: CREATE INDEXES FOR FASTER QUERIES
# ============================================================================

print("\n[4/6] Creating indexes for query optimization...")

cursor.execute("CREATE INDEX idx_galaxy_type ON galaxies(galaxy_type)")
cursor.execute("CREATE INDEX idx_odd_label ON galaxies(odd_label)")
cursor.execute("CREATE INDEX idx_consensus_anomaly ON anomaly_scores(consensus_anomaly)")
cursor.execute("CREATE INDEX idx_model_name ON ml_predictions(model_name)")

conn.commit()
print("✓ Created 4 indexes")

# ============================================================================
# STEP 4: EXAMPLE QUERIES
# ============================================================================

print("\n[5/6] Running example SQL queries...\n")

print("=" * 80)
print("QUERY 1: Galaxy type distribution")
print("=" * 80)

query1 = """
SELECT 
    galaxy_type_name,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM galaxies), 2) as percentage
FROM galaxies
GROUP BY galaxy_type_name
ORDER BY count DESC
"""

result1 = pd.read_sql_query(query1, conn)
print(result1.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 2: Top 10 most anomalous galaxies (consensus)")
print("=" * 80)

query2 = """
SELECT 
    g.galaxy_id,
    g.galaxy_type_name,
    g.smooth_fraction,
    g.disk_fraction,
    g.spiral_fraction,
    g.odd_fraction,
    a.pca_reconstruction_error,
    a.isolation_forest_score
FROM galaxies g
JOIN anomaly_scores a ON g.galaxy_id = a.galaxy_id
WHERE a.consensus_anomaly = 1
ORDER BY a.pca_reconstruction_error DESC
LIMIT 10
"""

result2 = pd.read_sql_query(query2, conn)
print(result2.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 3: Odd galaxies by type")
print("=" * 80)

query3 = """
SELECT 
    galaxy_type_name,
    COUNT(*) as total_galaxies,
    SUM(odd_label) as odd_galaxies,
    ROUND(SUM(odd_label) * 100.0 / COUNT(*), 2) as odd_percentage
FROM galaxies
GROUP BY galaxy_type_name
ORDER BY odd_percentage DESC
"""

result3 = pd.read_sql_query(query3, conn)
print(result3.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 4: Model performance comparison")
print("=" * 80)

query4 = """
SELECT 
    model_name,
    ROUND(accuracy, 4) as accuracy,
    ROUND(f1_macro, 4) as f1_score,
    ROUND(auroc, 4) as auroc
FROM model_performance
ORDER BY f1_macro DESC
"""

result4 = pd.read_sql_query(query4, conn)
print(result4.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 5: Clustering statistics")
print("=" * 80)

query5 = """
SELECT 
    c.kmeans_k3_cluster as cluster,
    COUNT(*) as galaxy_count,
    ROUND(AVG(g.smooth_fraction), 3) as avg_smooth,
    ROUND(AVG(g.disk_fraction), 3) as avg_disk,
    ROUND(AVG(g.spiral_fraction), 3) as avg_spiral
FROM clustering_results c
JOIN galaxies g ON c.galaxy_id = g.galaxy_id
GROUP BY c.kmeans_k3_cluster
ORDER BY cluster
"""

result5 = pd.read_sql_query(query5, conn)
print(result5.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 6: Anomaly detection method agreement")
print("=" * 80)

query6 = """
SELECT 
    SUM(isolation_forest_anomaly) as iso_forest_anomalies,
    SUM(lof_anomaly) as lof_anomalies,
    SUM(pca_anomaly) as pca_anomalies,
    SUM(consensus_anomaly) as consensus_anomalies,
    ROUND(SUM(consensus_anomaly) * 100.0 / SUM(isolation_forest_anomaly), 2) as agreement_rate
FROM anomaly_scores
"""

result6 = pd.read_sql_query(query6, conn)
print(result6.to_string(index=False))

print("\n" + "=" * 80)
print("QUERY 7: Spiral galaxies with high odd fractions")
print("=" * 80)

query7 = """
SELECT 
    galaxy_id,
    spiral_fraction,
    odd_fraction,
    CASE 
        WHEN odd_label = 1 THEN 'Yes'
        ELSE 'No'
    END as is_odd
FROM galaxies
WHERE galaxy_type_name = 'Spiral'
  AND odd_fraction > 0.5
ORDER BY odd_fraction DESC
LIMIT 10
"""

result7 = pd.read_sql_query(query7, conn)
print(result7.to_string(index=False))

# ============================================================================
# STEP 5: ADVANCED QUERIES WITH JOINS
# ============================================================================

print("\n[6/6] Running advanced queries with joins...\n")

print("=" * 80)
print("ADVANCED QUERY 1: Anomalous spiral galaxies")
print("=" * 80)

query_adv1 = """
SELECT 
    g.galaxy_id,
    g.spiral_fraction,
    g.disk_fraction,
    g.odd_fraction,
    a.pca_reconstruction_error,
    c.kmeans_k3_cluster
FROM galaxies g
JOIN anomaly_scores a ON g.galaxy_id = a.galaxy_id
JOIN clustering_results c ON g.galaxy_id = c.galaxy_id
WHERE g.galaxy_type_name = 'Spiral'
  AND a.consensus_anomaly = 1
ORDER BY a.pca_reconstruction_error DESC
LIMIT 10
"""

result_adv1 = pd.read_sql_query(query_adv1, conn)
print(result_adv1.to_string(index=False))

print("\n" + "=" * 80)
print("ADVANCED QUERY 2: Cluster purity analysis")
print("=" * 80)

query_adv2 = """
SELECT 
    c.kmeans_k3_cluster,
    g.galaxy_type_name,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY c.kmeans_k3_cluster), 2) as pct_in_cluster
FROM clustering_results c
JOIN galaxies g ON c.galaxy_id = g.galaxy_id
GROUP BY c.kmeans_k3_cluster, g.galaxy_type_name
ORDER BY c.kmeans_k3_cluster, count DESC
"""

result_adv2 = pd.read_sql_query(query_adv2, conn)
print(result_adv2.to_string(index=False))

print("\n" + "=" * 80)
print("ADVANCED QUERY 3: Best models by metric")
print("=" * 80)

query_adv3 = """
SELECT 
    'Highest Accuracy' as metric,
    model_name,
    ROUND(accuracy, 4) as value
FROM model_performance
WHERE accuracy = (SELECT MAX(accuracy) FROM model_performance)
UNION ALL
SELECT 
    'Highest F1 (Macro)' as metric,
    model_name,
    ROUND(f1_macro, 4) as value
FROM model_performance
WHERE f1_macro = (SELECT MAX(f1_macro) FROM model_performance)
UNION ALL
SELECT 
    'Highest AUROC' as metric,
    model_name,
    ROUND(auroc, 4) as value
FROM model_performance
WHERE auroc = (SELECT MAX(auroc) FROM model_performance)
"""

result_adv3 = pd.read_sql_query(query_adv3, conn)
print(result_adv3.to_string(index=False))

# ============================================================================
# STEP 6: SAVE QUERY RESULTS AND CREATE VISUALIZATIONS
# ============================================================================

print("\n\nSaving query results to CSV...")

result1.to_csv('outputs/sql/galaxy_type_distribution.csv', index=False)
result2.to_csv('outputs/sql/top_anomalies.csv', index=False)
result3.to_csv('outputs/sql/odd_galaxies_by_type.csv', index=False)
result4.to_csv('outputs/sql/model_performance.csv', index=False)
result5.to_csv('outputs/sql/clustering_statistics.csv', index=False)
result_adv2.to_csv('outputs/sql/cluster_purity.csv', index=False)
result_adv3.to_csv('outputs/sql/best_models.csv', index=False)

print("✓ Saved 7 query result files")

# Create visualization of cluster purity
print("\nCreating cluster purity visualization...")

fig, ax = plt.subplots(figsize=(12, 6))

pivot_data = result_adv2.pivot(index='kmeans_k3_cluster', 
                                columns='galaxy_type_name', 
                                values='pct_in_cluster').fillna(0)

pivot_data.plot(kind='bar', stacked=True, ax=ax, 
                color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_xlabel('K-Means Cluster', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Cluster Purity: Galaxy Type Distribution per Cluster', 
             fontsize=14, fontweight='bold')
ax.legend(title='Galaxy Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('outputs/sql/cluster_purity.png', dpi=300, bbox_inches='tight')
print("✓ Saved cluster_purity.png")
plt.close()

# Create model performance comparison visualization
print("Creating model performance comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models_df = pd.read_sql_query("SELECT * FROM model_performance ORDER BY f1_macro DESC", conn)

# Accuracy
axes[0].barh(models_df['model_name'], models_df['accuracy'], color='steelblue', edgecolor='black')
axes[0].set_xlabel('Accuracy', fontsize=11)
axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0].set_xlim([0, 1.05])
for i, v in enumerate(models_df['accuracy']):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# F1 Macro
axes[1].barh(models_df['model_name'], models_df['f1_macro'], color='coral', edgecolor='black')
axes[1].set_xlabel('F1 (Macro)', fontsize=11)
axes[1].set_title('Model F1 Score Comparison', fontsize=13, fontweight='bold')
axes[1].set_xlim([0, 1.05])
for i, v in enumerate(models_df['f1_macro']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# AUROC
axes[2].barh(models_df['model_name'], models_df['auroc'], color='seagreen', edgecolor='black')
axes[2].set_xlabel('AUROC', fontsize=11)
axes[2].set_title('Model AUROC Comparison', fontsize=13, fontweight='bold')
axes[2].set_xlim([0, 1.05])
for i, v in enumerate(models_df['auroc']):
    axes[2].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/sql/model_comparison_full.png', dpi=300, bbox_inches='tight')
print("✓ Saved model_comparison_full.png")
plt.close()

# ============================================================================
# CLEANUP
# ============================================================================

# Get database statistics
cursor.execute("SELECT COUNT(*) FROM galaxies")
galaxy_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM clustering_results")
clustering_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM anomaly_scores")
anomaly_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM model_performance")
model_count = cursor.fetchone()[0]

conn.close()

print("\n" + "=" * 80)
print("SQL INTEGRATION COMPLETE!")
print("=" * 80)
print(f"\nDatabase: galaxy_morphology.db")
print(f"   Galaxies: {galaxy_count:,}")
print(f"   Clustering results: {clustering_count:,}")
print(f"   Anomaly scores: {anomaly_count:,}")
print(f"   Model records: {model_count}")

print("\nGenerated files in 'outputs/sql/':")
print("   - galaxy_morphology.db (SQLite database)")
print("   - galaxy_type_distribution.csv")
print("   - top_anomalies.csv")
print("   - odd_galaxies_by_type.csv")
print("   - model_performance.csv")
print("   - clustering_statistics.csv")
print("   - cluster_purity.csv")
print("   - best_models.csv")
print("   - cluster_purity.png")
print("   - model_comparison_full.png")
print("=" * 80)