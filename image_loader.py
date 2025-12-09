import os
import pandas as pd
from PIL import Image
from datasets import load_from_disk

# -----------------------------------------------------
# Resolve absolute path to gz2_prepared
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "gz2_prepared")


def load_gz2_dataset(limit=5000):
    """
    Load the HuggingFace GZ2 dataset saved via save_to_disk()
    Convert to pandas DataFrame for Streamlit usage
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"gz2_prepared not found at {DATA_DIR}")

    print(f"Loading GZ2 dataset from disk: {DATA_DIR}")
    ds = load_from_disk(DATA_DIR)

    # Convert to DataFrame (limit for performance)
    df = ds.select(range(min(limit, len(ds)))).to_pandas()
    return df


def get_image(galaxy_id):
    """
    Load image from gz2_prepared/images/<galaxy_id>.jpg
    """
    img_path = os.path.join(DATA_DIR, "images", f"{galaxy_id}.jpg")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    return Image.open(img_path).convert("RGB")
