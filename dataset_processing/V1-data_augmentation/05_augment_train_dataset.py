"""Build the v1 enhanced dataset: copy the split, then augment train images.

Applies gamma correction, CLAHE, or brightness/contrast adjustment to the
training set only, at a higher probability for minority classes.
"""
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    YOLO_ROOT, ENHANCED_ROOT, CLASS_NAMES,
    IMAGE_SUFFIXES, SEED
)

MINORITY_CLASSES = {"Y.T.Marten", "Dog", "Cow"}  # fewest samples in the raw dataset
NORMAL_PROB = 0.30
MINORITY_PROB = 0.80  # augment minority classes more aggressively to offset imbalance


def gamma_adjust(image):
    gamma = random.uniform(0.6, 0.9)
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image, table)


def clahe_adjust(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def brightness_contrast(image):
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


METHODS = {
    "gamma": gamma_adjust,
    "clahe": clahe_adjust,
    "brightness": brightness_contrast,
}


def label_classes(label_path):
    ids = {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return {CLASS_NAMES[i] for i in ids}


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if ENHANCED_ROOT.exists():
        shutil.rmtree(ENHANCED_ROOT)

    stats = Counter()

    for split in ["train", "val", "test"]:
        src_images = YOLO_ROOT / "images" / split
        src_labels = YOLO_ROOT / "labels" / split
        dst_images = ENHANCED_ROOT / "images" / split
        dst_labels = ENHANCED_ROOT / "labels" / split
        dst_images.mkdir(parents=True)
        dst_labels.mkdir(parents=True)

        for image_path in sorted(src_images.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            label_path = src_labels / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dst_images / image_path.name)
            shutil.copy2(label_path, dst_labels / label_path.name)

            if split != "train":
                continue  # never augment val/test -- keeps evaluation unbiased

            classes = label_classes(label_path)
            prob = MINORITY_PROB if classes & MINORITY_CLASSES else NORMAL_PROB

            if random.random() >= prob:
                continue

            method_name = random.choice(list(METHODS))
            image = cv2.imread(str(image_path))
            augmented = METHODS[method_name](image)

            new_stem = f"{image_path.stem}_aug_{method_name}"
            cv2.imwrite(str(dst_images / f"{new_stem}{image_path.suffix}"), augmented)
            shutil.copy2(label_path, dst_labels / f"{new_stem}.txt")

            stats["augmented"] += 1
            stats[method_name] += 1
            stats["minority"] += int(bool(classes & MINORITY_CLASSES))

    for name in ["classes.txt", "class_mapping.json"]:
        shutil.copy2(YOLO_ROOT / name, ENHANCED_ROOT / name)

    data_yaml = f"""path: {ENHANCED_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        data_yaml += f"  {i}: {name}\n"

    (ENHANCED_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    report = f"""Augmentation Report
Augmented train images: {stats['augmented']}
Minority-class images augmented: {stats['minority']}
Gamma: {stats['gamma']}
CLAHE: {stats['clahe']}
Brightness/contrast: {stats['brightness']}
"""
    (ENHANCED_ROOT / "augmentation_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print(f"Output: {ENHANCED_ROOT}")


if __name__ == "__main__":
    main()
