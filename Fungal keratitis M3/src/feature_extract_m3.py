"""
Feature Extraction for Model M3 — Intensity Profile Features
=============================================================
Features extracted from local maxima intensity profiles:
  1. increasing_ramp_height  — mean rise from local mins to local maxes
  2. decreasing_ramp_height  — mean drop from local maxes to local mins
  3. top_width               — mean width of peaks at half prominence
  4. peak_width              — mean full width of peaks
  5. peak_height             — mean height of peaks (local max values)
  6. num_peaks               — number of significant local maxima
  7. mean_peak_prominence    — mean peak prominence
  8. mean_valley_depth       — mean depth of valleys

Total: 8 features

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m3.csv
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

FEATURE_NAMES = [
    "increasing_ramp_height", "decreasing_ramp_height",
    "top_width", "peak_width", "peak_height",
    "num_peaks", "mean_peak_prominence", "mean_valley_depth"
]


def extract_intensity_profile_features(img):
    """
    Extract intensity profile features from local maxima analysis.
    Takes horizontal and vertical intensity profiles, finds peaks.
    """
    h, w = img.shape

    # Sample multiple scan lines for robustness (middle rows and columns)
    row_indices = np.linspace(h // 4, 3 * h // 4, 5, dtype=int)
    col_indices = np.linspace(w // 4, 3 * w // 4, 5, dtype=int)

    all_peaks_heights = []
    all_prominences = []
    all_widths = []
    all_inc_ramps = []
    all_dec_ramps = []
    all_valley_depths = []

    profiles = []
    for r in row_indices:
        profiles.append(img[r, :].astype(float))
    for c in col_indices:
        profiles.append(img[:, c].astype(float))

    for profile in profiles:
        # Smooth profile slightly to reduce noise
        kernel = np.ones(5) / 5
        smoothed = np.convolve(profile, kernel, mode='same')

        peaks, properties = find_peaks(smoothed, height=10, distance=10,
                                       prominence=5, width=3)

        if len(peaks) == 0:
            continue

        heights = properties['peak_heights']
        prominences = properties['prominences']
        widths = properties['widths']

        all_peaks_heights.extend(heights.tolist())
        all_prominences.extend(prominences.tolist())
        all_widths.extend(widths.tolist())

        # Compute ramp heights and valley depths
        for i, pk in enumerate(peaks):
            left_base = int(properties['left_bases'][i])
            right_base = int(properties['right_bases'][i])
            inc_ramp = smoothed[pk] - smoothed[left_base]
            dec_ramp = smoothed[pk] - smoothed[right_base]
            all_inc_ramps.append(inc_ramp)
            all_dec_ramps.append(dec_ramp)

        # Valleys (local minima)
        valleys, _ = find_peaks(-smoothed, distance=10, prominence=5)
        if len(valleys) > 0:
            valley_vals = smoothed[valleys]
            all_valley_depths.extend((np.mean(smoothed) - valley_vals).tolist())

    # Aggregate
    def safe_mean(lst, default=0.0):
        return float(np.mean(lst)) if len(lst) > 0 else default

    increasing_ramp = safe_mean(all_inc_ramps)
    decreasing_ramp = safe_mean(all_dec_ramps)
    top_width = safe_mean(all_widths) * 0.5  # half-prominence width approximation
    peak_width = safe_mean(all_widths)
    peak_height = safe_mean(all_peaks_heights)
    num_peaks = float(len(all_peaks_heights))
    mean_prominence = safe_mean(all_prominences)
    mean_valley = safe_mean(all_valley_depths)

    return [increasing_ramp, decreasing_ramp, top_width, peak_width,
            peak_height, num_peaks, mean_prominence, mean_valley]


def process_class(class_name, label):
    src_dir = os.path.join(AUGMENTED_DIR, class_name)
    filenames = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    rows = []
    for fname in tqdm(filenames, desc=f"  {class_name}"):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        features = extract_intensity_profile_features(img)
        rows.append([fname, label] + features)
    return rows


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_path = os.path.join(FEATURES_DIR, "features_m3.csv")

    print("Model M3 — Feature Extraction (Intensity Profile Features)")
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
