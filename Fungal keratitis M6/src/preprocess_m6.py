"""
M6 Preprocessing: Background Noise Removal + MA Extraction + NMS + Region Growing
Specialized preprocessing for microaneurysm detection from IVCM images
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


def remove_background_noise(image):
    """
    Remove background noise using morphological operations
    
    Args:
        image: Input image
        
    Returns:
        Image with background noise removed
    """
    result = image.copy().astype(np.uint8)
    
    # Opening operation to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened


def extract_candidate_ma(image):
    """
    Extract candidate microaneurysm regions
    
    Args:
        image: Input image
        
    Returns:
        Candidate MA map
    """
    # Apply threshold to identify dark regions (potential MAs)
    _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create candidate map
    ma_map = np.zeros_like(image, dtype=np.float32)
    
    # Filter contours by area (MAs are small)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 500:  # Size range for MAs
            cv2.drawContours(ma_map, [contour], 0, 255, -1)
    
    return ma_map.astype(np.uint8)


def non_maximum_suppression(image, window_size=5):
    """
    Apply non-maximum suppression to remove duplicate detections
    
    Args:
        image: Input image with candidate regions
        window_size: Size of the suppression window
        
    Returns:
        Image after NMS
    """
    result = image.copy().astype(np.float32)
    
    # Apply max filter
    struct = ndimage.generate_binary_structure(2, 2)
    max_filtered = ndimage.maximum_filter(result, footprint=struct, size=window_size)
    
    # Keep only local maxima
    nms_result = (result == max_filtered).astype(np.float32) * result
    
    return nms_result


def region_growing(image, seed_threshold=150):
    """
    Apply region growing to connect candidate regions
    
    Args:
        image: Input image
        seed_threshold: Threshold for seeds
        
    Returns:
        Image after region growing
    """
    result = image.copy().astype(np.uint8)
    
    # Find seeds (local maxima)
    struct = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(result, footprint=struct)
    seeds = (result == local_max).astype(np.uint8)
    
    # Label seeds
    labeled, num_features = ndimage.label(seeds)
    
    # Grow regions from seeds
    grown = np.zeros_like(result)
    for label_id in range(1, num_features + 1):
        seed_mask = (labeled == label_id)
        
        # Dilate seed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(seed_mask.astype(np.uint8), kernel, iterations=2)
        
        # Use original image to limit growth
        grown[dilated > 0] = result[dilated > 0]
    
    return grown


def preprocess_m6(image_path):
    """
    Apply M6 preprocessing for MA-focused analysis
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = read_image(image_path)
    
    # Extract green channel (primary channel for IVCM)
    green_channel = img[:, :, 1].astype(np.uint8)
    
    # Step 1: Background Noise Removal
    denoised = remove_background_noise(green_channel)
    
    # Step 2: Candidate MA Extraction
    ma_candidates = extract_candidate_ma(denoised)
    
    # Step 3: Non-Maximum Suppression
    nms_result = non_maximum_suppression(ma_candidates, window_size=3)
    nms_result = nms_result.astype(np.uint8)
    
    # Step 4: Region Growing
    grown = region_growing(nms_result)
    
    # Combine original denoised image with extracted regions
    result = cv2.add(denoised, grown // 2)
    
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
            preprocessed = preprocess_m6(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, preprocessed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M6"""
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
    
    print("\n✓ M6 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
