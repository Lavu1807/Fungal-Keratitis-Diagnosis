"""
Feature Extraction for Model M1 — Green Channel
================================================
12 shape features extracted from binary contours:
  1. Area
  2. Perimeter
  3. Circularity  (4π·Area / Perimeter²)
  4. Length-to-width ratio (major_axis / minor_axis)
  5. Eccentricity
  6. Solidity  (Area / ConvexArea)
  7. Extent  (Area / BoundingRectArea)
  8. Convex area
  9. Major axis length
  10. Minor axis length
  11. Orientation (angle of the major axis)
  12. Equivalent diameter (sqrt(4·Area/π))

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m1.csv
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
    "area", "perimeter", "circularity", "length_to_width_ratio",
    "eccentricity", "solidity", "extent", "convex_area",
    "major_axis", "minor_axis", "orientation", "equivalent_diameter"
]


def extract_shape_features(img):
    """
    Extract 12 shape features from a grayscale image.
    Uses Otsu thresholding to get binary mask, then finds the largest contour.
    """
    # Threshold to get binary mask
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0.0] * 12

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Circularity
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Fit ellipse (needs at least 5 points)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (ma_len, mi_len), angle = ellipse
        major_axis = max(ma_len, mi_len)
        minor_axis = min(ma_len, mi_len)
        orientation = angle
    else:
        major_axis = 0.0
        minor_axis = 0.0
        orientation = 0.0

    # Length-to-width ratio
    lw_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    # Eccentricity: sqrt(1 - (minor/major)^2)
    if major_axis > 0:
        eccentricity = math.sqrt(1 - (minor_axis / major_axis) ** 2)
    else:
        eccentricity = 0.0

    # Solidity
    hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(hull)
    solidity = area / convex_area if convex_area > 0 else 0.0

    # Extent
    x, y, w, h = cv2.boundingRect(contour)
    bounding_area = w * h
    extent = area / bounding_area if bounding_area > 0 else 0.0

    # Equivalent diameter
    equivalent_diameter = math.sqrt(4 * area / math.pi) if area > 0 else 0.0

    return [
        area, perimeter, circularity, lw_ratio,
        eccentricity, solidity, extent, convex_area,
        major_axis, minor_axis, orientation, equivalent_diameter
    ]


def process_class(class_name, label):
    """Extract features for all images in a class directory."""
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
    output_path = os.path.join(FEATURES_DIR, "features_m1.csv")

    print("Model M1 — Feature Extraction (12 Shape Features)")
    print("=" * 55)

    all_rows = []
    all_rows.extend(process_class("FK", 1))
    all_rows.extend(process_class("normal", 0))

    # Write CSV
    header = ["filename", "label"] + FEATURE_NAMES
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\nDone! {len(all_rows)} samples × {len(FEATURE_NAMES)} features")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
