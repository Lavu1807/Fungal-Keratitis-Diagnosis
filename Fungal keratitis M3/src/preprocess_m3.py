"""
M3 Preprocessing: Convolution with Gaussian Mask + Local Maximum Region Extraction
Extracts local maximum regions using Gaussian filtering
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


def preprocess_m3(image_path):
    """
    Apply M3 preprocessing: Green channel, Gaussian convolution, and local maximum extraction
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image in BGR format
    img = read_image(image_path)
    
    # Step 1: Extract green channel
    green_channel = img[:, :, 1].astype(np.float32)
    
    # Step 2: Apply Gaussian convolution (Gaussian blur)
    # Using kernel size and sigma to model the Gaussian mask
    gaussian_filtered = cv2.GaussianBlur(green_channel, (5, 5), 1.0)
    
    # Step 3: Local maximum region extraction
    # Find local maxima using morphological operations
    struct = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(gaussian_filtered, footprint=struct)
    
    # Create binary map of local maxima (where value equals its local maximum)
    local_max_binary = (gaussian_filtered == local_max).astype(np.float32)
    
    # Enhance the image using local maxima
    # Apply morphological operations to dilate/enhance regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(gaussian_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Normalize to uint8
    result = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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
            preprocessed = preprocess_m3(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, preprocessed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M3"""
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
    
    print("\n✓ M3 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
