# Night Wildlife Detection Dataset

## 1. Dataset Source

The original dataset was collected using infrared camera traps in the **Northeast Tiger and Leopard National Park**.

The complete public dataset contains **25,657 images** covering **17 animal categories**, with annotations provided in **Pascal VOC XML** format.

For this project, **only the nighttime subset** was used.

Nighttime dataset statistics:

- Images: **10,344**
- XML annotations: **10,344**
- Objects: **10,699**
- Classes: **17**
- Invalid boxes (out-of-bounds or degenerate, clipped during conversion): **1,233**
- Main image resolutions:
  - 1280 × 720
  - 1600 × 1200

---

## 2. Preprocessing Pipeline

The complete preprocessing workflow builds three progressively-enhanced dataset versions from the same raw source and split:

```text
Raw VOC Dataset (datasets/raw-dataset/)
        │
        ▼
Dataset Analysis
        │
        ▼
VOC → YOLO Conversion  ──────────────────►  datasets/yolo_dataset/
        │
        ▼
Train / Validation / Test Split (70/20/10, seeded)
        │
        ▼
Training Image Augmentation (v1: gamma / CLAHE / brightness-contrast)
        │                                    │
        │                                    ▼
        │                          datasets/enhanced_yolo_dataset/
        ▼
Dataset Verification
        │
        ▼
BBox-aware CutOut Augmentation (v2, built on top of v1)
        │
        ▼
Dataset Verification (v2)
        │
        ▼
datasets/enhanced_yolo_dataset_v2/
```

The preprocessing scripts live in three subfolders under `dataset_processing/`, each importing shared paths/constants from `config.py` one level up:

| Folder | Script | Description |
|---|---|---|
| `raw_data-to-yolo_format/` | `01_analyze_voc_dataset.py` | Analyze the raw VOC dataset and report statistics. |
| `raw_data-to-yolo_format/` | `02_convert_voc_to_yolo.py` | Convert Pascal VOC XML annotations into YOLO TXT format. |
| `raw_data-to-yolo_format/` | `03_split_dataset.py` | Split the dataset into Train / Validation / Test sets using a fixed random seed. |
| `raw_data-to-yolo_format/` | `04_analyze_split_distribution.py` | Check per-class distribution after splitting. |
| `V1-data_augmentation/` | `05_augment_train_dataset.py` | Build the v1 dataset: photometric augmentation (gamma / CLAHE / brightness-contrast) on the training set only. |
| `V1-data_augmentation/` | `06_verify_yolo_dataset.py` | Verify the integrity of the v1 dataset. |
| `V2-data_augmentation/` | `07_augment_v2_bbox_cutout.py` | Build the v2 dataset: bounding-box-aware CutOut augmentation on top of v1, targeting the classes the baseline model struggled with most. |
| `V2-data_augmentation/` | `08_verify_v2_yolo_dataset.py` | Verify the integrity of the v2 dataset. |

---

## 3. Dataset Split

The raw dataset was split before any augmentation, stratified by class:

| Split | Images |
|---|---:|
| Train | 7,233 |
| Validation | 2,061 |
| Test | 1,050 |

Split ratio: **~70% train / 20% validation / 10% test**, grouped by each image's lowest class id present so every class gets a proportional split rather than the dataset being split as one undifferentiated pool.

The random seed (`SEED = 42`) is fixed, so the split is reproducible by running the scripts in order.

---

## 4. v1 — Photometric Augmentation (Low-Light Enhancement)

Applied **only to the training set**; validation and test sets are copied unchanged to keep evaluation unbiased.

Three photometric methods, chosen at random per augmented image:

- Gamma Correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Brightness / Contrast Adjustment

No geometric transformations (rotation, flip, crop, scale) are applied, so the original YOLO labels remain valid and are copied directly alongside each augmented image.

### Class Imbalance Strategy

The raw dataset has relatively few samples for `Y.T.Marten`, `Dog`, and `Cow`. These minority classes are augmented far more aggressively:

| Image type | Augmentation probability |
|---|---:|
| Minority-class images | 80% |
| Other training images | 30% |

### v1 Augmentation Summary

| Item | Value |
|---|---:|
| Original training images | 7,233 |
| Augmented training images | 2,321 |
| Final training images | 9,554 |
| Minority-class images augmented | 201 |
| Gamma | 787 |
| CLAHE | 755 |
| Brightness / Contrast | 779 |

### v1 Final Dataset Statistics

| Split | Images | Objects |
|---|---:|---:|
| Train | 9,554 | 9,888 |
| Validation | 2,061 | 2,131 |
| Test | 1,050 | 1,080 |
| **Total** | **12,665** | **13,099** |

Verification: **PASSED** — 0 missing labels, 0 orphan labels, 0 empty labels, 0 invalid labels.

---

## 5. v2 — BBox-aware CutOut Augmentation (Occlusion Robustness)

Built **on top of v1** (`enhanced_yolo_dataset` is copied wholesale, then additional images are added to the training set only). Targets the classes the v1 baseline scored lowest on: `Dog`, `RoeDeer`, `Badger`, `Y.T.Marten`.

For a selected training image, a small patch inside one object's bounding box is masked with that region's median color (never a jarring solid block), capped so the masked area never exceeds 20% of the box — the goal is partial occlusion robustness, not destroying the object.

| Image type | CutOut probability |
|---|---:|
| Images containing a hard class | 60% |
| Other training images | 15% |

Already-augmented (`_aug_`) or already-cutout (`_cutout_`) images are skipped as source candidates, so augmentations don't stack on top of each other.

### v2 Augmentation Summary

| Item | Value |
|---|---:|
| New CutOut images | 1,425 |
| Final training images | 10,979 |

Target class distribution (new CutOut images only):

| Class | CutOut images |
|---|---:|
| RoeDeer | 214 |
| Badger | 158 |
| Hare | 134 |
| MuskDeer | 122 |
| RaccoonDog | 114 |
| WildBoar | 86 |
| RedFox | 90 |
| SikaDeer | 73 |
| Leopard | 55 |
| Weasel | 53 |
| Dog | 52 |
| Y.T.Marten | 52 |
| Sable | 51 |
| LeopardCat | 78 |
| BlackBear | 37 |
| AmurTiger | 42 |
| Cow | 14 |

### v2 Final Dataset Statistics

| Split | Images | Objects |
|---|---:|---:|
| Train | 10,979 | 11,378 |
| Validation | 2,061 | 2,131 |
| Test | 1,050 | 1,080 |
| **Total** | **14,090** | **14,589** |

Verification: **PASSED** — 0 missing labels, 0 orphan labels, 0 empty labels, 0 invalid labels.

---

## 6. Directory Structure

```text
datasets/
├── raw-dataset/                  # original Pascal VOC source
│   ├── Annotations/
│   └── JPEGImages/
├── yolo_dataset/                 # raw -> YOLO format, split, no augmentation
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── data.yaml
│   ├── classes.txt
│   └── class_mapping.json
├── enhanced_yolo_dataset/        # v1: + photometric augmentation
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── data.yaml
│   ├── classes.txt
│   ├── class_mapping.json
│   ├── augmentation_report.txt
│   └── final_dataset_verification_report.txt
└── enhanced_yolo_dataset_v2/     # v2: + bbox-aware CutOut augmentation
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── data.yaml
    ├── classes.txt
    ├── class_mapping.json
    ├── augmentation_v2_report.txt
    └── v2_dataset_verification_report.txt
```

`datasets/` is not tracked in git (14GB) — regenerate it locally by running the scripts below in order.

---

## 7. YOLO Label Format

Each image has a corresponding label file with the same stem:

```text
images/train/000001.jpg
labels/train/000001.txt
```

Each line in a label file follows the YOLO format:

```text
class_id x_center y_center width height
```

Example:

```text
5 0.462500 0.371875 0.182812 0.241667
```

All coordinates are normalized to the range **[0, 1]**.

---

## 8. How to Reproduce

Install dependencies (from the project root):

```bash
pip install -r requirements.txt
```

Then run the scripts in order. Each script resolves paths via `dataset_processing/config.py`, so run them from within their own folder:

```bash
cd dataset_processing/raw_data-to-yolo_format
python 01_analyze_voc_dataset.py
python 02_convert_voc_to_yolo.py
python 03_split_dataset.py
python 04_analyze_split_distribution.py

cd ../V1-data_augmentation
python 05_augment_train_dataset.py
python 06_verify_yolo_dataset.py

cd ../V2-data_augmentation
python 07_augment_v2_bbox_cutout.py
python 08_verify_v2_yolo_dataset.py
```

To train, point Ultralytics YOLO at whichever version's `data.yaml` you want:

```text
datasets/yolo_dataset/data.yaml                # raw
datasets/enhanced_yolo_dataset/data.yaml        # v1
datasets/enhanced_yolo_dataset_v2/data.yaml     # v2
```

No additional preprocessing or format conversion is required.
