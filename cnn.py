"""
Galaxy Morphology Analysis - Step 4: Shallow CNN
Train a shallow CNN (3-4 conv layers) on galaxy images
Compare performance with classical ML models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class GalaxyDataset(Dataset):
    def __init__(self, dataset, indices, labels, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels  # numpy array of labels
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.dataset[int(real_idx)]
        
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[real_idx]
        return image, label

# ============================================================================
# SHALLOW CNN ARCHITECTURE
# ============================================================================

class ShallowCNN(nn.Module):
    """
    Shallow CNN with 4 convolutional layers
    Simple architecture to keep training fast and interpretable
    """
    def __init__(self, num_classes=3):
        super(ShallowCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling layers: 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_cnn():
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    os.makedirs("outputs/cnn", exist_ok=True)
    
    print("=" * 80)
    print("SHALLOW CNN FOR GALAXY MORPHOLOGY CLASSIFICATION")
    print("=" * 80)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("\n[1/8] Loading dataset with images...")
    ds = load_from_disk("gz2_prepared")
    print(f"✓ Loaded {len(ds):,} galaxies with images")
    
    # Check image size
    sample_img = ds[0]['image']
    print(f"   Image size: {sample_img.size}")
    print(f"   Image mode: {sample_img.mode}")
    
    # Create target labels
    def create_target(row):
        if row['spiral_label'] == 1:
            return 2  # Spiral
        elif row['disk_label'] == 1:
            return 1  # Disk
        else:
            return 0  # Smooth
    
    print("\n[2/8] Creating target labels...")
    df = ds.to_pandas()
    df['galaxy_type'] = df.apply(create_target, axis=1)
    
    print("   Class distribution:")
    for cls, name in enumerate(['Smooth', 'Disk', 'Spiral']):
        count = (df['galaxy_type'] == cls).sum()
        pct = (count / len(df)) * 100
        print(f"     {name}: {count:,} ({pct:.1f}%)")
    
    # ========================================================================
    # STEP 2: STRATIFIED TRAIN/VAL/TEST SPLIT
    # ========================================================================
    print("\n[3/8] Creating stratified train/val/test split...")
    
    # First split: 80% train, 20% test
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df['galaxy_type']
    )
    
    # Second split: 80% train, 20% val (from the 80%)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=42,
        stratify=df.iloc[train_idx]['galaxy_type']
    )
    
    print(f"   Train: {len(train_idx):,} samples")
    print(f"   Val:   {len(val_idx):,} samples")
    print(f"   Test:  {len(test_idx):,} samples")
    
    # ========================================================================
    # STEP 3: CREATE DATA TRANSFORMS AND LOADERS
    # ========================================================================
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for val/test
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    print("\n[4/8] Creating data loaders...")
    
    # Convert labels to numpy array for easy indexing
    all_labels = df['galaxy_type'].values
    
    train_dataset = GalaxyDataset(ds, train_idx, all_labels, transform=train_transform)
    val_dataset = GalaxyDataset(ds, val_idx, all_labels, transform=val_test_transform)
    test_dataset = GalaxyDataset(ds, test_idx, all_labels, transform=val_test_transform)
    
    BATCH_SIZE = 64
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    print(f"✓ Created data loaders (batch size: {BATCH_SIZE})")
    
    # ========================================================================
    # STEP 4: BUILD MODEL
    # ========================================================================
    
    print("\n[5/8] Building Shallow CNN architecture...")
    model = ShallowCNN(num_classes=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # STEP 5: TRAINING SETUP
    # ========================================================================
    
    print("\n[6/8] Setting up training...")
    
    # Compute class weights
    class_counts = df['galaxy_type'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    NUM_EPOCHS = 15
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Optimizer: Adam (lr=0.001)")
    
    # ========================================================================
    # STEP 6: TRAINING LOOP
    # ========================================================================
    
    print("\n[7/8] Training CNN...")
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "outputs/cnn/best_model.pth")
        
        print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        print(f"   Best Val F1: {best_val_f1:.4f}\n")
    
    print(f"\n✓ Training complete! Best Val F1: {best_val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # ========================================================================
    # STEP 7: EVALUATION ON TEST SET
    # ========================================================================
    
    print("\n[8/8] Evaluating on test set...")
    
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # Compute metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    test_auroc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    
    print("\n" + "=" * 80)
    print("CNN TEST SET PERFORMANCE")
    print("=" * 80)
    print(f"Accuracy:        {test_acc:.4f}")
    print(f"F1 (Macro):      {test_f1_macro:.4f}")
    print(f"F1 (Weighted):   {test_f1_weighted:.4f}")
    print(f"AUROC (OvR):     {test_auroc:.4f}")
    
    print("\nPer-class Performance:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Smooth', 'Disk', 'Spiral'],
                              digits=4))
    
    # Save results
    cnn_results = pd.DataFrame([{
        'Model': 'Shallow CNN',
        'Accuracy': test_acc,
        'F1 (Macro)': test_f1_macro,
        'F1 (Weighted)': test_f1_weighted,
        'AUROC (OvR)': test_auroc
    }])
    cnn_results.to_csv("outputs/cnn/cnn_results.csv", index=False)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print("\nCreating visualizations...")
    
    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, NUM_EPOCHS + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score (Macro)', fontsize=12)
    axes[2].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/cnn/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training_curves.png")
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=['Smooth', 'Disk', 'Spiral'],
                yticklabels=['Smooth', 'Disk', 'Spiral'],
                cbar_kws={'label': 'Normalized'})
    plt.title(f'CNN Confusion Matrix\nF1 (Macro): {test_f1_macro:.4f}', 
             fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/cnn/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confusion_matrix.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("SHALLOW CNN TRAINING COMPLETE!")
    print("=" * 80)
    print("\nGenerated files in 'outputs/cnn/':")
    print("   - best_model.pth")
    print("   - cnn_results.csv")
    print("   - training_curves.png")
    print("   - confusion_matrix.png")
    print("=" * 80)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    train_cnn()