"""
Download Galaxy Zoo 2 dataset from Hugging Face
"""
from datasets import load_dataset
import os

print("=" * 70)
print("DOWNLOADING GALAXY ZOO 2 DATASET")
print("=" * 70)
print("\nThis will download ~2GB of data...")
print("Please be patient, this may take 5-15 minutes.\n")

# Download dataset
print("Downloading from Hugging Face...")
dataset = load_dataset("mwalmsley/gz2")

# Save to disk
print("\nSaving to disk as 'gz2_prepared'...")
dataset['train'].save_to_disk("gz2_prepared")

print("\n✓ Download complete!")
print(f"Dataset location: {os.path.abspath('gz2_prepared')}")
print(f"Total galaxies: {len(dataset['train']):,}")
print("\nYou can now run the analysis pipeline:")
print("  python 1_eda.py")
