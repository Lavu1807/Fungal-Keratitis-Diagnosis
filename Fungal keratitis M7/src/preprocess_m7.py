"""
M7 Preprocessing: Illumination + Adaptive Histogram Equalization + Greyscale Normalization + MA Candidate Extraction
Advanced preprocessing focusing on local contrast and MA detection
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


def apply_illumination_equalization(image):
    """
    Apply illumination equalization using Gaussian kernel
    
    Args:
        image: Input image
        
    Returns:
        Illumination equalized image
    """
    # Apply CLAHE for illumination equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image.astype(np.uint8))
    
    return equalized.astype(np.float32)


def apply_adaptive_histogram_equalization(image):
    """
    Apply adaptive histogram equalization
    
    Args:
        image: Input image
        
    Returns:
        Adaptively equalized image
    """
    # Convert to uint8 if needed
    img_uint8 = image.astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    result = clahe.apply(img_uint8)
    
    return result.astype(np.float32)


def greyscale_normalization(image):
    """
    Apply greyscale normalization (min-max normalization)
    
    Args:
        image: Input image
        
    Returns:
        Normalized greyscale image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val > min_val:
        normalized = ((image - min_val) / (max_val - min_val)) * 255
    else:
        normalized = image
    
    return normalized


def extract_ma_candidates(image):
    """
    Extract microaneurysm candidates using morphological operations
    
    Args:
        image: Preprocessed image
        
    Returns:
        MA candidate regions highlighted
    """
    # Convert to uint8
    img_uint8 = image.astype(np.uint8)
    
    # Use morphological operations to detect dark spots (MAs)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Morphological top-hat to extract small objects
    tophat = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    
    # Use black-hat to detect dark objects
    blackhat = cv2.morphologyEx(img_uint8, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine both (MAs can appear as dark regions)
    # Use blackhat which detects dark objects
    candidates = blackhat
    
    # Apply threshold to identify strong candidates
    _, ma_mask = cv2.threshold(candidates, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to connect and clean
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ma_mask = cv2.morphologyEx(ma_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    return ma_mask


def preprocess_m7(image_path):
    """
    Apply M7 preprocessing: Illumination + Adaptive Histogram + Greyscale Norm + MA Extraction
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = read_image(image_path)
    
    # Extract green channel
    green_channel = img[:, :, 1].astype(np.float32)
    
    # Step 1: Illumination Equalization
    illumination_eq = apply_illumination_equalization(green_channel)
    
    # Step 2: Adaptive Histogram Equalization
    adaptive_he = apply_adaptive_histogram_equalization(illumination_eq)
    
    # Step 3: Greyscale Normalization
    normalized = greyscale_normalization(adaptive_he)
    
    # Step 4: Extract MA Candidates
    ma_candidates = extract_ma_candidates(normalized)
    
    # Combine normalized image with MA candidate regions
    # Scale down MA candidates to avoid overwhelming the original
    combined = normalized.astype(np.uint8)
    combined = cv2.add(combined, ma_candidates // 3)
    
    # Final normalization
    result = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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
            preprocessed = preprocess_m7(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, preprocessed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M7"""
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
    
    print("\n✓ M7 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
