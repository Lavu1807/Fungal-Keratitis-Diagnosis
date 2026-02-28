"""
M5 Preprocessing: Green Plane + Median Filtering + CLAHE + Vessel Removal + Mathematical Morphology
Advanced preprocessing with vessel removal using mathematical morphology
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


def remove_vessels(image):
    """
    Remove vessel structures using morphological operations
    
    Args:
        image: Input image
        
    Returns:
        Image with vessels removed
    """
    # Create a copy
    result = image.copy().astype(np.uint8)
    
    # Use morphological opening to remove thin structures (vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Use morphological closing to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    return result


def apply_mathematical_morphology(image):
    """
    Apply mathematical morphology operations (gradient and reconstruction)
    
    Args:
        image: Input image
        
    Returns:
        Morphologically enhanced image
    """
    result = image.copy().astype(np.uint8)
    
    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
    
    # Add gradient back for edge enhancement
    result = cv2.add(result, gradient)
    
    return result


def preprocess_m5(image_path):
    """
    Apply M5 preprocessing: Green plane, median filtering, CLAHE, vessel removal, morphology
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = read_image(image_path)
    
    # Step 1: Extract green plane/channel
    green_plane = img[:, :, 1].astype(np.float32)
    
    # Step 2: Median Filtering (denoising)
    median_filtered = cv2.medianBlur(green_plane.astype(np.uint8), 5)
    median_filtered = median_filtered.astype(np.float32)
    
    # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(median_filtered.astype(np.uint8))
    clahe_applied = clahe_applied.astype(np.float32)
    
    # Step 4: Vessel Removal
    vessel_removed = remove_vessels(clahe_applied)
    
    # Step 5: Mathematical Morphology
    morphology_applied = apply_mathematical_morphology(vessel_removed)
    
    # Final normalization
    result = cv2.normalize(morphology_applied, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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
            preprocessed = preprocess_m5(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, preprocessed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M5"""
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
    
    print("\n✓ M5 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
