"""
Feature Extraction for Model M7 — Cross-section Profile + Local Saliency
=========================================================================
Features based on cross-section profiles, local transformations, and saliency:

Cross-section profile features (4):
  1. mean_cross_section_peak    — mean peak value across multiple profile slices
  2. mean_cross_section_width   — mean FWHM of cross-section peaks
  3. cross_section_std          — std dev of cross-section intensities
  4. cross_section_skew         — skewness of cross-section intensities

Local cross-section transformation (4):
  5. local_gradient_mean        — mean gradient magnitude
  6. local_gradient_std         — std of gradient magnitude
  7. local_laplacian_mean       — mean Laplacian response
  8. local_laplacian_std        — std of Laplacian response

Local saliency analysis (4):
  9. saliency_mean              — mean saliency map value
  10. saliency_std              — std of saliency map
  11. saliency_peak_count       — number of salient regions
  12. saliency_coverage         — fraction of image above saliency threshold

Total: 12 features

Input:  data/augmented/FK/ and data/augmented/normal/
Output: data/features/features_m7.csv
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import skew

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

FEATURE_NAMES = [
    # Cross-section profile
    "mean_cross_section_peak", "mean_cross_section_width",
    "cross_section_std", "cross_section_skew",
    # Local cross-section transformation
    "local_gradient_mean", "local_gradient_std",
    "local_laplacian_mean", "local_laplacian_std",
    # Local saliency analysis
    "saliency_mean", "saliency_std",
    "saliency_peak_count", "saliency_coverage"
]


def extract_cross_section_features(img):
    """Extract cross-section intensity profile features."""
    h, w = img.shape
    img_f = img.astype(np.float64)

    # Sample multiple horizontal and vertical profiles
    row_indices = np.linspace(h // 4, 3 * h // 4, 7, dtype=int)
    col_indices = np.linspace(w // 4, 3 * w // 4, 7, dtype=int)

    peaks_vals = []
    widths_vals = []
    all_intensities = []

    for r in row_indices:
        profile = img_f[r, :]
        all_intensities.extend(profile.tolist())
        pks, props = find_peaks(profile, height=10, distance=8, width=2)
        if len(pks) > 0:
            peaks_vals.extend(props['peak_heights'].tolist())
            widths_vals.extend(props['widths'].tolist())

    for c in col_indices:
        profile = img_f[:, c]
        all_intensities.extend(profile.tolist())
        pks, props = find_peaks(profile, height=10, distance=8, width=2)
        if len(pks) > 0:
            peaks_vals.extend(props['peak_heights'].tolist())
            widths_vals.extend(props['widths'].tolist())

    mean_peak = float(np.mean(peaks_vals)) if peaks_vals else 0.0
    mean_width = float(np.mean(widths_vals)) if widths_vals else 0.0
    cs_std = float(np.std(all_intensities)) if all_intensities else 0.0
    cs_skew = float(skew(all_intensities)) if len(all_intensities) > 2 else 0.0

    return [mean_peak, mean_width, cs_std, cs_skew]


def extract_local_transformation_features(img):
    """Extract local gradient and Laplacian features."""
    img_f = img.astype(np.float64)

    # Gradient magnitude (Sobel)
    grad_x = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # Laplacian
    laplacian = cv2.Laplacian(img_f, cv2.CV_64F, ksize=3)
    lap_mean = float(np.mean(np.abs(laplacian)))
    lap_std = float(np.std(laplacian))

    return [grad_mean, grad_std, lap_mean, lap_std]


def extract_saliency_features(img):
    """Extract saliency-based features using spectral residual method."""
    img_f = img.astype(np.float64)

    # Spectral residual saliency
    # 1. FFT
    fft = np.fft.fft2(img_f)
    fft_shift = np.fft.fftshift(fft)

    # 2. Log amplitude and phase
    amplitude = np.abs(fft_shift)
    phase = np.angle(fft_shift)
    log_amplitude = np.log1p(amplitude)

    # 3. Spectral residual = log_amplitude - smoothed_log_amplitude
    kernel = np.ones((3, 3)) / 9.0
    smoothed_log = cv2.filter2D(log_amplitude, -1, kernel)
    spectral_residual = log_amplitude - smoothed_log

    # 4. Reconstruct saliency map
    saliency_complex = np.exp(spectral_residual + 1j * phase)
    saliency_spatial = np.fft.ifft2(np.fft.ifftshift(saliency_complex))
    saliency_map = np.abs(saliency_spatial) ** 2

    # Normalize saliency map to [0, 1]
    sal_min, sal_max = saliency_map.min(), saliency_map.max()
    if sal_max > sal_min:
        saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)
    else:
        saliency_map = np.zeros_like(saliency_map)

    # Smooth
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 2.5)

    sal_mean = float(np.mean(saliency_map))
    sal_std = float(np.std(saliency_map))

    # Salient regions: threshold at mean + 2*std
    threshold = sal_mean + 2 * sal_std
    salient_mask = (saliency_map > threshold).astype(np.uint8) * 255

    # Count connected components as peak count
    contours, _ = cv2.findContours(salient_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sal_peak_count = float(len(contours))

    # Coverage
    sal_coverage = float(np.sum(saliency_map > threshold)) / (img.shape[0] * img.shape[1])

    return [sal_mean, sal_std, sal_peak_count, sal_coverage]


def downscale(img, max_dim=256):
    """Downscale image if larger than max_dim for speed."""
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return img


def extract_all_features(img):
    small = downscale(img, 256)
    cs = extract_cross_section_features(small)
    lt = extract_local_transformation_features(small)
    sal = extract_saliency_features(small)
    return cs + lt + sal


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
    output_path = os.path.join(FEATURES_DIR, "features_m7.csv")

    print("Model M7 — Feature Extraction (12 Features: Cross-section+Gradient+Saliency)")
    print("=" * 75)

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
