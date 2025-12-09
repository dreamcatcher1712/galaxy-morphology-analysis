import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

print("=" * 80)
print("CLASSICAL ML MODELS - IMPROVED WITH PROPER IMBALANCE HANDLING")
print("=" * 80)

# Load dataset
print("\n[1/10] Loading preprocessed dataset...")
ds = load_from_disk("gz2_prepared")
df = ds.to_pandas()
print(f"✓ Loaded {len(df):,} galaxies")

# Feature engineering
print("\n[2/10] Engineering features...")
fraction_cols = [c for c in df.columns if c.endswith("_fraction")]
print(f"   Using {len(fraction_cols)} fraction features")

# Create target variable: multi-class classification
def create_target(row):
    if row['spiral_label'] == 1:
        return 2  # Spiral
    elif row['disk_label'] == 1:
        return 1  # Disk
    else:
        return 0  # Smooth/Elliptical

df['galaxy_type'] = df.apply(create_target, axis=1)

# Check class distribution
print("\n   Target class distribution:")
class_counts = df['galaxy_type'].value_counts().sort_index()
class_names = {0: 'Smooth', 1: 'Disk', 2: 'Spiral'}
for cls, count in class_counts.items():
    pct = (count / len(df)) * 100
    print(f"     {class_names[cls]} ({cls}): {count:,} ({pct:.1f}%)")

# Calculate class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(df['galaxy_type']), 
                                     y=df['galaxy_type'])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\n   Computed class weights: {class_weight_dict}")

# Prepare features and target
print("\n[3/10] Preparing STRATIFIED train/test split...")
X = df[fraction_cols].fillna(0)
y = df['galaxy_type']

# STRATIFIED split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # ✓ Stratified
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")
print("\n   Class distribution in train set:")
for cls, count in y_train.value_counts().sort_index().items():
    pct = (count / len(y_train)) * 100
    print(f"     {class_names[cls]}: {count:,} ({pct:.1f}%)")

# Standardize features
print("\n[4/10] Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "outputs/models/scaler.pkl")

# Handle class imbalance with Random Undersampling + Class Weights
# SMOTE is too memory-intensive for this dataset, so we use undersampling instead
print("\n[5/10] Handling class imbalance with Random Undersampling...")
print("   Note: Undersampling majority class to reduce imbalance")

from imblearn.under_sampling import RandomUnderSampler

# Define sampling strategy: keep all minority, reduce majority to 2x the spiral class
sampling_strategy = {
    0: 70000,  # Smooth: reduce from ~97k to 70k
    1: 6648,   # Disk: keep all (minority)
    2: 33807   # Spiral: keep all
}

rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)

try:
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
    print(f"   ✓ Undersampling completed successfully")
    print(f"   Original training size: {len(X_train_scaled):,}")
    print(f"   After undersampling: {len(X_train_resampled):,}")
    print("\n   Class distribution after undersampling:")
    for cls, count in pd.Series(y_train_resampled).value_counts().sort_index().items():
        pct = (count / len(y_train_resampled)) * 100
        print(f"     {class_names[cls]}: {count:,} ({pct:.1f}%)")
    print("\n   ✓ Will also use class weights in models for extra robustness")
except Exception as e:
    print(f"   ⚠ Undersampling failed: {e}")
    print("   Continuing without resampling - using class weights only")
    X_train_resampled = X_train_scaled
    y_train_resampled = y_train

# Dictionary to store results
results = {}

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION (with class weights)
# ============================================================================
print("\n[6/10] Training Logistic Regression with class weights...")
lr_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    multi_class='ovr',
    class_weight='balanced'  # ✓ Handle imbalance
)
lr_model.fit(X_train_resampled, y_train_resampled)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)

results['Logistic Regression'] = {
    'model': lr_model,
    'predictions': lr_pred,
    'probabilities': lr_pred_proba
}
print(f"   ✓ Trained")
joblib.dump(lr_model, "outputs/models/logistic_regression.pkl")

# ============================================================================
# MODEL 2: RANDOM FOREST (with class weights)
# ============================================================================
print("\n[7/10] Training Random Forest with class weights...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20, 
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # ✓ Handle imbalance
)
rf_model.fit(X_train_resampled, y_train_resampled)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)

results['Random Forest'] = {
    'model': rf_model,
    'predictions': rf_pred,
    'probabilities': rf_pred_proba
}
print(f"   ✓ Trained")
joblib.dump(rf_model, "outputs/models/random_forest.pkl")

# ============================================================================
# MODEL 3: SVM (with class weights, smaller sample)
# ============================================================================
print("\n[8/10] Training SVM with class weights (on stratified 30% sample)...")
# Use stratified sampling to ensure all classes are represented
from sklearn.model_selection import train_test_split as split_sample

# Take 30% stratified sample for SVM (to ensure all classes present)
X_train_svm, _, y_train_svm, _ = split_sample(
    X_train_resampled, 
    y_train_resampled, 
    train_size=0.3,  # 30% of resampled data
    random_state=42,
    stratify=y_train_resampled  # Stratify to keep all classes
)

print(f"   SVM training on {len(X_train_svm):,} samples")
print(f"   Class distribution in SVM sample:")
for cls, count in pd.Series(y_train_svm).value_counts().sort_index().items():
    print(f"     {class_names[cls]}: {count:,}")

svm_model = SVC(
    kernel='rbf', 
    random_state=42, 
    probability=True,
    class_weight='balanced'  # ✓ Handle imbalance
)
svm_model.fit(X_train_svm, y_train_svm)
svm_pred = svm_model.predict(X_test_scaled)
svm_pred_proba = svm_model.predict_proba(X_test_scaled)

results['SVM'] = {
    'model': svm_model,
    'predictions': svm_pred,
    'probabilities': svm_pred_proba
}
print(f"   ✓ Trained")
joblib.dump(svm_model, "outputs/models/svm.pkl")

# ============================================================================
# COMPREHENSIVE EVALUATION WITH PROPER METRICS
# ============================================================================
print("\n[9/10] Computing comprehensive evaluation metrics...")

# Calculate all metrics for each model
metrics_data = []
for model_name, res in results.items():
    pred = res['predictions']
    pred_proba = res['probabilities']
    
    # Basic metrics
    accuracy = accuracy_score(y_test, pred)
    
    # Per-class metrics
    f1_macro = f1_score(y_test, pred, average='macro')
    f1_weighted = f1_score(y_test, pred, average='weighted')
    precision_macro = precision_score(y_test, pred, average='macro')
    recall_macro = recall_score(y_test, pred, average='macro')
    
    # AUROC (One-vs-Rest for multi-class)
    try:
        auroc_ovr = roc_auc_score(y_test, pred_proba, multi_class='ovr', average='macro')
    except:
        auroc_ovr = 0.0
    
    metrics_data.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 (Macro)': f1_macro,
        'F1 (Weighted)': f1_weighted,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'AUROC (OvR)': auroc_ovr
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values('F1 (Macro)', ascending=False)
metrics_df.to_csv("outputs/results/comprehensive_metrics.csv", index=False)

print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL PERFORMANCE (Sorted by F1 Macro)")
print("=" * 80)
print(metrics_df.to_string(index=False))

# Detailed per-class reports
print("\n" + "=" * 80)
print("PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)

for model_name, res in results.items():
    print(f"\n{model_name}:")
    print("-" * 80)
    report = classification_report(
        y_test, 
        res['predictions'],
        target_names=['Smooth', 'Disk', 'Spiral'],
        digits=4
    )
    print(report)
    
    with open(f"outputs/results/{model_name.lower().replace(' ', '_')}_report.txt", 'w') as f:
        f.write(report)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[10/10] Creating advanced visualizations...")

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (model_name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['predictions'])
    # Normalize by true label
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Smooth', 'Disk', 'Spiral'],
                yticklabels=['Smooth', 'Disk', 'Spiral'],
                cbar_kws={'label': 'Normalized'})
    f1 = metrics_df[metrics_df['Model'] == model_name]['F1 (Macro)'].values[0]
    axes[idx].set_title(f'{model_name}\nF1 (Macro): {f1:.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/results/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrices.png")
plt.close()

# 2. Metrics Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart comparison
metrics_to_plot = ['F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 
                   'Recall (Macro)', 'AUROC (OvR)']
x = np.arange(len(metrics_to_plot))
width = 0.25

for idx, model_name in enumerate(results.keys()):
    model_metrics = metrics_df[metrics_df['Model'] == model_name][metrics_to_plot].values[0]
    axes[0].bar(x + idx*width, model_metrics, width, label=model_name)

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Metrics Comparison Across Models', fontsize=14, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1])

# F1 Score comparison
models = metrics_df['Model'].tolist()
f1_scores = metrics_df['F1 (Macro)'].tolist()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = axes[1].barh(models, f1_scores, color=colors, edgecolor='black')
axes[1].set_xlabel('F1 Score (Macro)', fontsize=12)
axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlim([0, 1])
axes[1].grid(axis='x', alpha=0.3)

for bar, score in zip(bars, f1_scores):
    axes[1].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/results/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved metrics_comparison.png")
plt.close()

# 3. ROC Curves (One-vs-Rest)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for model_idx, (model_name, res) in enumerate(results.items()):
    pred_proba = res['probabilities']
    
    for class_idx in range(3):
        # Binarize the output
        y_test_binary = (y_test == class_idx).astype(int)
        y_score = pred_proba[:, class_idx]
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        axes[model_idx].plot(fpr, tpr, lw=2, 
                            label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})')
    
    axes[model_idx].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    axes[model_idx].set_xlim([0.0, 1.0])
    axes[model_idx].set_ylim([0.0, 1.05])
    axes[model_idx].set_xlabel('False Positive Rate', fontsize=11)
    axes[model_idx].set_ylabel('True Positive Rate', fontsize=11)
    axes[model_idx].set_title(f'{model_name} - ROC Curves', fontsize=12, fontweight='bold')
    axes[model_idx].legend(loc="lower right")
    axes[model_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/results/roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved roc_curves.png")
plt.close()

# 4. Precision-Recall Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for model_idx, (model_name, res) in enumerate(results.items()):
    pred_proba = res['probabilities']
    
    for class_idx in range(3):
        y_test_binary = (y_test == class_idx).astype(int)
        y_score = pred_proba[:, class_idx]
        
        precision, recall, _ = precision_recall_curve(y_test_binary, y_score)
        avg_precision = average_precision_score(y_test_binary, y_score)
        
        axes[model_idx].plot(recall, precision, lw=2,
                            label=f'{class_names[class_idx]} (AP = {avg_precision:.3f})')
    
    axes[model_idx].set_xlim([0.0, 1.0])
    axes[model_idx].set_ylim([0.0, 1.05])
    axes[model_idx].set_xlabel('Recall', fontsize=11)
    axes[model_idx].set_ylabel('Precision', fontsize=11)
    axes[model_idx].set_title(f'{model_name} - PR Curves', fontsize=12, fontweight='bold')
    axes[model_idx].legend(loc="lower left")
    axes[model_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved precision_recall_curves.png")
plt.close()

# 5. Feature Importance (Random Forest)
print("   Creating feature importance plot...")
feature_importance = pd.DataFrame({
    'Feature': fraction_cols,
    'Importance': results['Random Forest']['model'].feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['Importance'], 
         color='steelblue', edgecolor='black')
plt.yticks(range(len(feature_importance)), 
           feature_importance['Feature'].str.replace('_fraction', '').str.replace('-', ' '))
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved feature_importance.png")
plt.close()