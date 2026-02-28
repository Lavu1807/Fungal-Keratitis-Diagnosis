"""
M1 Preprocessing: Green Channel Extraction
Extracts green channel from images for fungal keratitis classification
"""
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
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


def preprocess_m1(image_path):
    """
    Extract green channel from image
    
    Args:
        image_path: Path to input image
        
    Returns:
        Green channel as numpy array
    """
    # Read image in BGR format
    img = read_image(image_path)
    
    # Extract green channel (index 1 in BGR format)
    green_channel = img[:, :, 1]
    
    return green_channel


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
            green_channel = preprocess_m1(img_path)
            
            # Save processed image
            output_path = os.path.join(processed_dir, img_path.name)
            cv2.imwrite(output_path, green_channel)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Main preprocessing pipeline for M1"""
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
    
    print("\n✓ M1 Preprocessing Complete!")
    print(f"Processed FK images saved to: {processed_fk_dir}")
    print(f"Processed normal images saved to: {processed_normal_dir}")


if __name__ == "__main__":
    main()
