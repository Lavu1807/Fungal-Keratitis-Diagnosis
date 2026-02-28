"""
Feature Extraction for Model M6 — Shape + Intensity + SBF Features
===================================================================
Features from MA (microaneurysm) candidate regions:

Shape-based (5):
  1. ma_count          — number of MA candidate regions
  2. mean_ma_area      — mean area of MA candidates
  3. mean_ma_perimeter — mean perimeter of MA candidates
  4. mean_ma_circularity — mean circularity of MA candidates
  5. total_ma_area     — total area of all MA candidates

Intensity-based (4):
  6. mean_intensity    — mean pixel intensity of full image
  7. std_intensity     — std dev of pixel intensity
  8. min_intensity     — minimum intensity
  9. max_intensity     — maximum intensity

SBF-based (Spatial Binary Features) (4):
  10. sbf_edge_density    — edge pixel density (Canny)
  11. sbf_texture_energy  — energy from GLCM-like local variance
  12. sbf_boundary_ratio  — boundary pixels / total foreground pixels
  13. sbf_compactness     — mean compactness of regions

Total: 13 features

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m6.csv
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
    # Shape-based
    "ma_count", "mean_ma_area", "mean_ma_perimeter",
    "mean_ma_circularity", "total_ma_area",
    # Intensity-based
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
    # SBF-based
    "sbf_edge_density", "sbf_texture_energy",
    "sbf_boundary_ratio", "sbf_compactness"
]


def extract_features(img):
    """Extract shape, intensity, and SBF features."""
    h, w = img.shape
    total_pixels = h * w

    # ── Shape-based: find MA candidates via thresholding ──
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small candidate regions (MA-like: area between 5 and 2000 pixels)
    ma_candidates = [c for c in contours if 5 <= cv2.contourArea(c) <= 2000]

    ma_count = float(len(ma_candidates))
    if len(ma_candidates) > 0:
        areas = [cv2.contourArea(c) for c in ma_candidates]
        perimeters = [cv2.arcLength(c, True) for c in ma_candidates]
        circularities = [(4 * math.pi * a) / (p ** 2) if p > 0 else 0.0
                         for a, p in zip(areas, perimeters)]
        mean_ma_area = float(np.mean(areas))
        mean_ma_perimeter = float(np.mean(perimeters))
        mean_ma_circularity = float(np.mean(circularities))
        total_ma_area = float(np.sum(areas))
    else:
        mean_ma_area = 0.0
        mean_ma_perimeter = 0.0
        mean_ma_circularity = 0.0
        total_ma_area = 0.0

    # ── Intensity-based ──
    img_float = img.astype(np.float64)
    mean_intensity = float(np.mean(img_float))
    std_intensity = float(np.std(img_float))
    min_intensity = float(np.min(img_float))
    max_intensity = float(np.max(img_float))

    # ── SBF-based features ──
    # Edge density (Canny)
    edges = cv2.Canny(img, 50, 150)
    sbf_edge_density = float(np.sum(edges > 0)) / total_pixels

    # Texture energy (local variance)
    local_mean = cv2.blur(img_float, (5, 5))
    local_var = cv2.blur((img_float - local_mean) ** 2, (5, 5))
    sbf_texture_energy = float(np.mean(local_var))

    # Boundary ratio
    foreground = np.sum(binary > 0)
    if foreground > 0:
        # Erode to find interior, boundary = foreground - interior
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=1)
        interior = np.sum(eroded > 0)
        boundary = foreground - interior
        sbf_boundary_ratio = float(boundary) / float(foreground)
    else:
        sbf_boundary_ratio = 0.0

    # Compactness: mean of (perimeter² / area) for all contours
    if len(ma_candidates) > 0:
        compactness_vals = [(cv2.arcLength(c, True) ** 2) / cv2.contourArea(c)
                           if cv2.contourArea(c) > 0 else 0.0
                           for c in ma_candidates]
        sbf_compactness = float(np.mean(compactness_vals))
    else:
        sbf_compactness = 0.0

    return [
        ma_count, mean_ma_area, mean_ma_perimeter,
        mean_ma_circularity, total_ma_area,
        mean_intensity, std_intensity, min_intensity, max_intensity,
        sbf_edge_density, sbf_texture_energy,
        sbf_boundary_ratio, sbf_compactness
    ]


def process_class(class_name, label):
    src_dir = os.path.join(AUGMENTED_DIR, class_name)
    filenames = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    rows = []
    for fname in tqdm(filenames, desc=f"  {class_name}"):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        features = extract_features(img)
        rows.append([fname, label] + features)
    return rows


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DIR, "features_m6.csv")

    print("Model M6 — Feature Extraction (13 Features: Shape+Intensity+SBF)")
    print("=" * 65)

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
