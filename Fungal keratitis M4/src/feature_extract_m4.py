"""
Feature Extraction for Model M4 — Shape Descriptors (Post Complex Pipeline)
===========================================================================
6 shape features from binary contour analysis:
  1. Relative area     — contour area / total image area
  2. Elongation        — 1 - (minor_axis / major_axis)
  3. Eccentricity      — sqrt(1 - (minor/major)²)
  4. Circularity       — 4π·Area / Perimeter²
  5. Rectangularity    — Area / BoundingRect area
  6. Solidity          — Area / ConvexHull area

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m4.csv
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

FEATURE_NAMES = [
    "relative_area", "elongation", "eccentricity",
    "circularity", "rectangularity", "solidity"
]


def extract_shape_features(img):
    """Extract 6 shape descriptor features."""
    total_area = img.shape[0] * img.shape[1]

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0.0] * 6

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Relative area
    relative_area = area / total_area if total_area > 0 else 0.0

    # Fit ellipse for elongation and eccentricity
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (ma_len, mi_len), angle = ellipse
        major_axis = max(ma_len, mi_len)
        minor_axis = min(ma_len, mi_len)
    else:
        major_axis = minor_axis = 0.0

    # Elongation: 1 - (minor/major)
    elongation = 1.0 - (minor_axis / major_axis) if major_axis > 0 else 0.0

    # Eccentricity
    eccentricity = math.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0

    # Circularity
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Rectangularity (extent)
    x, y, w, h = cv2.boundingRect(contour)
    rectangularity = area / (w * h) if (w * h) > 0 else 0.0

    # Solidity
    hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(hull)
    solidity = area / convex_area if convex_area > 0 else 0.0

    return [relative_area, elongation, eccentricity,
            circularity, rectangularity, solidity]


def process_class(class_name, label):
    src_dir = os.path.join(AUGMENTED_DIR, class_name)
    filenames = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    rows = []
    for fname in tqdm(filenames, desc=f"  {class_name}"):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        features = extract_shape_features(img)
        rows.append([fname, label] + features)
    return rows


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DIR, "features_m4.csv")

    print("Model M4 — Feature Extraction (6 Shape Descriptors)")
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
