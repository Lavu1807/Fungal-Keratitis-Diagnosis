"""
M4 Preprocessing: Comprehensive Multi-Step Preprocessing Pipeline
Pipeline (per Table 4 of the paper):
  1. Spatial resize (256×256)
  2. Illumination equalization (CLAHE)
  3. Denoising (bilateral filter)
  4. Adaptive contrast adjustment (local mean/std normalization)
  5. Colour normalization
  6. Optic disk removal
  7. Green channel extraction
  8. Dynamics enhancing (histogram stretching + gamma correction)
  9. Local maxima region extraction (regional maximum detection)
"""
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def read_image(image_path):
    """Read image with OpenCV, falling back to Pillow for unsupported formats."""
    img = cv2.imread(str(image_path))
    if img is None:
        try:
            pil_img = Image.open(str(image_path)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Could not read image: {image_path}")
    return img


def remove_optic_disk(image):
    """
    Estimate and remove optic disk region (bright center region)
    
    Args:
        image: Input image
        
    Returns:
        Image with optic disk removed
    """
    # Create a copy to work with
    result = image.copy().astype(np.float32)
    
    # Find bright regions (optic disk typically brighter)
    threshold = np.percentile(result, 90)
    bright_mask = result > threshold
    
    # Create a mask for optic disk (typically in center)
    h, w = result.shape
    center_y, center_x = h // 2, w // 2
    
    # Create circular region in center
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(h, w) // 4) ** 2
    
    # Find intersection of bright and center regions
    optic_disk_mask = bright_mask & mask
    
    # Replace optic disk with local mean
    if np.any(optic_disk_mask):
        mean_val = np.mean(result[~optic_disk_mask])
        result[optic_disk_mask] = mean_val
    
    return result


def dynamics_enhancing(image):
    """
    Enhance the dynamic range of the image.
    Uses percentile-based histogram stretching followed by gamma correction
    to reveal subtle structures in both dark and bright regions.
    """
    img = image.astype(np.float32)

    # Percentile-based histogram stretching (robust to outliers)
    p_low, p_high = np.percentile(img, (2, 98))
    if p_high > p_low:
        stretched = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    else:
        stretched = img / 255.0 if img.max() > 1 else img

    # Adaptive gamma correction (gamma < 1 brightens dark regions)
    mean_val = np.mean(stretched)
    gamma = np.log(0.5) / np.log(mean_val + 1e-8)   # targets mid-grey
    gamma = np.clip(gamma, 0.4, 2.5)                  # keep reasonable
    enhanced = np.power(stretched, gamma)

    return (enhanced * 255).astype(np.float32)


def local_maxima_region_extraction(image):
    """
    Extract local maxima regions from the image.
    Detects regional maxima using a local maximum filter, then creates
    an enhanced image that highlights these peak regions.
    """
    img = image.astype(np.float32)

    # Detect local maxima with a 3×3 neighbourhood
    struct = ndimage.generate_binary_structure(2, 2)          # 8-connected
    local_max = ndimage.maximum_filter(img, footprint=struct)
    local_max_mask = (img == local_max)                       # pixels that are local maxima

    # Dilate maxima to form small region patches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region_mask = cv2.dilate(
        local_max_mask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)

    # Combine: keep original where local-max regions exist, attenuate elsewhere
    result = img.copy()
    result[~region_mask] *= 0.7          # suppress non-peak background

    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


def preprocess_m4(image_path, target_size=(256, 256)):
    """
    Apply M4 preprocessing — full 9-step pipeline per the paper.
    """
    # Read image
    img = read_image(image_path)

    # Step 1: Spatial Resize
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Step 2: Green channel extraction
    green_channel = resized[:, :, 1].astype(np.float32)

    # Step 3: Denoising (bilateral filter)
    denoised = cv2.bilateralFilter(green_channel, 9, 75, 75)

    # Step 4: Illumination Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    illumination_eq = clahe.apply(denoised.astype(np.uint8)).astype(np.float32)

    # Step 5: Adaptive Contrast Adjustment (local mean/std)
    kernel_size = 31
    mean = cv2.blur(illumination_eq, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(illumination_eq ** 2, (kernel_size, kernel_size))
    std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))
    std = np.maximum(std, 1.0)
    adaptive_contrast = (illumination_eq - mean) / std

    # Step 6: Colour Normalization
    normalized = cv2.normalize(adaptive_contrast, None, 0, 255, cv2.NORM_MINMAX)

    # Step 7: Optic Disk Removal
    disk_removed = remove_optic_disk(normalized)

    # Step 8: Dynamics Enhancing
    enhanced = dynamics_enhancing(disk_removed)

    # Step 9: Local Maxima Region Extraction
    result = local_maxima_region_extraction(enhanced)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def process_dataset(raw_dir, processed_dir):
    """
    Process all images in raw directory and save to processed directory
    
    Args:
        raw_dir: Path to raw image directory
        processed_dir: Path to save processed images
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(Path(raw_dir).glob(f'*{ext}'))
        image_files.extend(Path(raw_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {raw_dir}")
    
    for img_path in tqdm(image_files, desc=f"Processing {os.path.basename(raw_dir)}"):
        try:
            # Preprocess image
            preprocessed = preprocess_m4(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, preprocessed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M4"""
    base_dir = Path(__file__).parent.parent
    
    # Process FK images
    raw_fk_dir = base_dir / 'data' / 'raw' / 'FK pic'
    processed_fk_dir = base_dir / 'data' / 'processed' / 'FK'
    
    print("\n=== Processing Fungal Keratitis Images ===")
    process_dataset(raw_fk_dir, processed_fk_dir)
    
    # Process Normal images
    raw_normal_dir = base_dir / 'data' / 'raw' / 'normal'
    processed_normal_dir = base_dir / 'data' / 'processed' / 'normal'
    
    print("\n=== Processing Normal Images ===")
    process_dataset(raw_normal_dir, processed_normal_dir)
    
    print("\n✓ M4 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
