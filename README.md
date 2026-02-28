# Ensemble Machine Learning Approaches for Automated Fungal Keratitis Diagnosis

Ensemble machine learning pipeline for diagnosing **Fungal Keratitis (FK)** from **In Vivo Confocal Microscopy (IVCM)** images. This project evaluates 9 classifiers across 7 distinct image-processing and feature-extraction pipelines (M1–M7).

> Based on the paper: *"Ensemble Machine Learning Approaches for Automated Fungal Keratitis Diagnosis Using In Vivo Confocal Microscopy Images"* — Healthcare Technology Letters.

---

## Project Structure

```
├── Fungal keratitis M1/   # Pipeline 1
├── Fungal keratitis M2/   # Pipeline 2
├── ...
├── Fungal keratitis M7/   # Pipeline 7
└── Ensemble_Machine_Learning_Approaches_for_Automated (1) (2).md   # Paper manuscript
```

Each pipeline folder (`M1`–`M7`) follows the same layout:

```
Fungal keratitis MN/
├── requirements.txt
├── data/
│   ├── raw/            # Original IVCM images (FK + normal)
│   ├── processed/      # Preprocessed images
│   ├── augmented/      # Augmented images (balanced classes)
│   ├── features/       # Extracted feature CSV
│   └── results/        # Classification result JSONs
└── src/
    ├── preprocess_mN.py
    ├── augment_mN.py
    ├── feature_extract_mN.py
    ├── classify_adaboost.py
    ├── classify_gnb.py
    ├── classify_knn.py
    ├── classify_lr.py
    ├── classify_moe.py
    ├── classify_rf.py
    ├── classify_rusboost.py
    ├── classify_snapshot.py
    └── classify_svc.py
```

## Classifiers

| Classifier | File |
|---|---|
| AdaBoost | `classify_adaboost.py` |
| Gaussian Naive Bayes | `classify_gnb.py` |
| K-Nearest Neighbors | `classify_knn.py` |
| Logistic Regression | `classify_lr.py` |
| Mixture of Experts | `classify_moe.py` |
| Random Forest | `classify_rf.py` |
| RUSBoost | `classify_rusboost.py` |
| Snapshot Ensemble | `classify_snapshot.py` |
| Support Vector Classifier | `classify_svc.py` |

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r "Fungal keratitis M1/requirements.txt"
```

> Each pipeline folder has its own `requirements.txt`. Install from the pipeline you intend to run.

### 3. Prepare data

Place your IVCM images in the appropriate `data/raw/` directory:

```
data/raw/FK pic/    ← Fungal keratitis images (.jpg)
data/raw/normal/    ← Normal images (.jpg)
```

## Usage

Run the pipeline steps **in order** from within a model folder:

```bash
cd "Fungal keratitis M1"

# Step 1: Preprocess raw images
python src/preprocess_m1.py

# Step 2: Augment images (balance classes)
python src/augment_m1.py

# Step 3: Extract features
python src/feature_extract_m1.py

# Step 4: Run a classifier
python src/classify_rf.py
```

Repeat for any pipeline (M1–M7) and any of the 9 classifiers.

## Data (Not Tracked by Git)

Image data and generated outputs are excluded from version control due to size (~55,000+ images). The following directories are gitignored:

- `**/data/raw/` — Original IVCM images
- `**/data/processed/` — Preprocessed images
- `**/data/augmented/` — Augmented images
- `**/data/features/` — Extracted feature CSVs
- `**/data/results/` — Classification result JSONs

To reproduce results, obtain the raw images and run the full pipeline as described above.

## License

See the accompanying paper for terms of use.
