"""
Feature Extraction for Model M5 — Intensity + DoG Features
===========================================================
8 features:
  1. mean_intensity    — mean pixel intensity
  2. std_intensity     — standard deviation of pixel intensity
  3-8. dog_1 to dog_6  — 6 Difference of Gaussian (DoG) responses
       at different sigma pairs: (1,2), (2,4), (3,6), (4,8), (5,10), (6,12)
       Each DoG value = mean absolute response of (G_sigma1 - G_sigma2)

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m5.csv
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

# Sigma pairs for Difference of Gaussian
DOG_SIGMAS = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)]

FEATURE_NAMES = [
    "mean_intensity", "std_intensity",
    "dog_1", "dog_2", "dog_3", "dog_4", "dog_5", "dog_6"
]


def compute_dog(img_float, sigma1, sigma2):
    """Compute Difference of Gaussian response."""
    # Kernel size must be odd and large enough for sigma
    ksize1 = int(6 * sigma1 + 1) | 1  # ensure odd
    ksize2 = int(6 * sigma2 + 1) | 1
    g1 = cv2.GaussianBlur(img_float, (ksize1, ksize1), sigma1)
    g2 = cv2.GaussianBlur(img_float, (ksize2, ksize2), sigma2)
    dog = g1 - g2
    return float(np.mean(np.abs(dog)))


def extract_features(img):
    """Extract intensity and DoG features."""
    img_float = img.astype(np.float32)

    mean_intensity = float(np.mean(img_float))
    std_intensity = float(np.std(img_float))

    # Downscale for DoG to speed up (preserve relative responses)
    h, w = img.shape
    if max(h, w) > 256:
        scale = 256.0 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        small = img_float

    dog_responses = []
    for s1, s2 in DOG_SIGMAS:
        dog_responses.append(compute_dog(small, s1, s2))

    return [mean_intensity, std_intensity] + dog_responses


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
    output_path = os.path.join(FEATURES_DIR, "features_m5.csv")

    print("Model M5 — Feature Extraction (Intensity + 6 DoG Responses)")
    print("=" * 60)

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
