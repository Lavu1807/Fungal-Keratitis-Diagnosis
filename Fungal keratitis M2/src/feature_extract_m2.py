"""Feature Extraction for Model M2 — Raw Intensity Histogram (Paper-Faithful)
==========================================================================
Features:
  Raw intensity histogram with 32 bins (unnormalized counts).
  No shape, colour, brightness, or contrast features.
  No post-histogram normalization.

Total: 32 features

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m2.csv
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

N_BINS = 32
FEATURE_NAMES = [f"hist_bin_{i}" for i in range(N_BINS)]


def extract_all_features(img):
    """Extract raw intensity histogram features (32 bins, unnormalized)."""
    hist = cv2.calcHist([img], [0], None, [N_BINS], [0, 256]).flatten()
    return hist.tolist()  # raw counts, no normalization


def process_class(class_name, label):
    src_dir = os.path.join(AUGMENTED_DIR, class_name)
    filenames = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    rows = []
    for fname in tqdm(filenames, desc=f"  {class_name}"):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        features = extract_all_features(img)
        rows.append([fname, label] + features)
    return rows


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DIR, "features_m2.csv")

    print("Model M2 — Feature Extraction (32 Raw Histogram Bins)")
    print("=" * 55)

    all_rows = []
    all_rows.extend(process_class("FK", 1))
    all_rows.extend(process_class("normal", 0))

    header = ["filename", "label"] + FEATURE_NAMES
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\nDone! {len(all_rows)} samples × {len(FEATURE_NAMES)} features")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
