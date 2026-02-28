"""
Data Augmentation for Model M1 (Green Channel Extraction)
=========================================================
Augmentation Strategy:
  - Horizontal flip, Vertical flip
  - Random rotation (±30°)
  - Random crop (80-95% of image, resized back)
  - Light Gaussian blur (kernel=3)
  - Extra augmentations on minority class (normal) to balance with FK

Input:  data/processed/FK/ and data/processed/normal/
Output: data/augmented/FK/ and data/augmented/normal/
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ── Model-specific augmentation parameters ──────────────────────
ROTATION_RANGE = (-30, 30)      # degrees
CROP_FACTOR_RANGE = (0.80, 0.95)  # fraction of original image to keep
GAUSSIAN_BLUR_KERNEL = 3        # 0 = disabled
RANDOM_SEED = 42

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")


def random_rotation(img, rng):
    """Rotate image by a random angle within ROTATION_RANGE."""
    angle = rng.uniform(*ROTATION_RANGE)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_crop(img, rng):
    """Crop a random region (CROP_FACTOR_RANGE of original) and resize back."""
    h, w = img.shape[:2]
    factor = rng.uniform(*CROP_FACTOR_RANGE)
    new_h, new_w = int(h * factor), int(w * factor)
    top = rng.integers(0, h - new_h + 1)
    left = rng.integers(0, w - new_w + 1)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def light_gaussian_blur(img):
    """Apply a light Gaussian blur."""
    if GAUSSIAN_BLUR_KERNEL > 0:
        return cv2.GaussianBlur(img, (GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_KERNEL), 0)
    return img


def generate_augmentations(img, rng, count):
    """
    Generate `count` augmented versions of `img`.
    Cycles through: h-flip, v-flip, rotation, crop, blur, and combinations.
    """
    augmented = []
    transforms = [
        lambda im: cv2.flip(im, 1),                         # horizontal flip
        lambda im: cv2.flip(im, 0),                         # vertical flip
        lambda im: random_rotation(im, rng),                 # random rotation
        lambda im: random_crop(im, rng),                     # random crop
        lambda im: light_gaussian_blur(im),                  # light blur
        lambda im: cv2.flip(random_rotation(im, rng), 1),   # h-flip + rotation
        lambda im: random_crop(random_rotation(im, rng), rng),  # rotation + crop
        lambda im: light_gaussian_blur(cv2.flip(im, 0)),    # v-flip + blur
        lambda im: random_rotation(random_crop(im, rng), rng),  # crop + rotation
        lambda im: cv2.flip(im, -1),                         # h+v flip (180°)
    ]
    for i in range(count):
        t = transforms[i % len(transforms)]
        augmented.append(t(img.copy()))
    return augmented


def augment_class(class_name, target_total, rng):
    """
    Augment images for a given class to reach `target_total` images.
    Copies originals + generates augmented images.
    """
    src_dir = os.path.join(PROCESSED_DIR, class_name)
    dst_dir = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(dst_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    n_originals = len(filenames)

    if n_originals == 0:
        print(f"  WARNING: No images found in {src_dir}")
        return 0

    n_augmented_needed = max(0, target_total - n_originals)
    # Distribute augmentations evenly across originals
    aug_per_image = n_augmented_needed // n_originals
    extra = n_augmented_needed % n_originals

    saved = 0
    print(f"  {class_name}: {n_originals} originals → target {target_total} "
          f"(+{n_augmented_needed} augmented)")

    for idx, fname in enumerate(tqdm(filenames, desc=f"  Augmenting {class_name}")):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Save original
        base, ext = os.path.splitext(fname)
        cv2.imwrite(os.path.join(dst_dir, fname), img)
        saved += 1

        # How many augmentations for this image
        n_aug = aug_per_image + (1 if idx < extra else 0)
        if n_aug > 0:
            augs = generate_augmentations(img, rng, n_aug)
            for j, aug_img in enumerate(augs):
                aug_fname = f"{base}_aug{j}{ext}"
                cv2.imwrite(os.path.join(dst_dir, aug_fname), aug_img)
                saved += 1

    return saved


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Count originals
    fk_files = [f for f in os.listdir(os.path.join(PROCESSED_DIR, "FK"))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    normal_files = [f for f in os.listdir(os.path.join(PROCESSED_DIR, "normal"))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    n_fk = len(fk_files)
    n_normal = len(normal_files)

    print(f"Model M1 — Data Augmentation")
    print(f"{'=' * 50}")
    print(f"Original counts: FK={n_fk}, Normal={n_normal}")
    print(f"Augmentation params: rotation={ROTATION_RANGE}, "
          f"crop={CROP_FACTOR_RANGE}, blur_kernel={GAUSSIAN_BLUR_KERNEL}")

    # Compute targets:
    # FK gets 2x augmentation (common augments for diversity)
    # Normal gets enough to match FK's augmented count (balance)
    fk_target = n_fk * 3       # ~3x originals (original + 2 augmented per image)
    normal_target = fk_target   # match FK count for class balance

    print(f"Targets: FK={fk_target}, Normal={normal_target}")
    print()

    fk_saved = augment_class("FK", fk_target, rng)
    normal_saved = augment_class("normal", normal_target, rng)

    print()
    print(f"Done! FK: {fk_saved} images, Normal: {normal_saved} images")
    print(f"Saved to: {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()
