"""Build the v2 dataset: add bounding-box-aware CutOut augmentation on top of v1.

Masks a small patch inside an object's bounding box with the region's median
color, applied more often on classes the baseline model struggled with.
"""
from pathlib import Path
from collections import Counter
import random
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "datasets"
V1_ROOT = ROOT / "enhanced_yolo_dataset"
V2_ROOT = ROOT / "enhanced_yolo_dataset_v2"

CLASS_NAMES = [
    "RaccoonDog", "Hare", "MuskDeer", "LeopardCat", "RedFox",
    "WildBoar", "SikaDeer", "RoeDeer", "AmurTiger", "Weasel",
    "Leopard", "Sable", "BlackBear", "Badger", "Y.T.Marten",
    "Dog", "Cow",
]

HARD_CLASSES = {"Dog", "RoeDeer", "Badger", "Y.T.Marten"}  # classes the v1 baseline scored lowest on
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

SEED = 42
NORMAL_CUTOUT_PROB = 0.15
HARD_CLASS_CUTOUT_PROB = 0.60
MAX_OCCLUSION_RATIO = 0.20  # cap how much of the box can be masked, so the object stays recognizable


def read_yolo_labels(label_path):
    objects = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, x, y, w, h = line.split()
        objects.append(
            {
                "class_id": int(class_id),
                "x_center": float(x),
                "y_center": float(y),
                "width": float(w),
                "height": float(h),
            }
        )

    return objects


def yolo_to_xyxy(obj, image_width, image_height):
    box_width = obj["width"] * image_width
    box_height = obj["height"] * image_height
    center_x = obj["x_center"] * image_width
    center_y = obj["y_center"] * image_height

    xmin = int(center_x - box_width / 2)
    ymin = int(center_y - box_height / 2)
    xmax = int(center_x + box_width / 2)
    ymax = int(center_y + box_height / 2)

    xmin = max(0, min(xmin, image_width - 1))
    ymin = max(0, min(ymin, image_height - 1))
    xmax = max(1, min(xmax, image_width))
    ymax = max(1, min(ymax, image_height))

    return xmin, ymin, xmax, ymax


def apply_bbox_cutout(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    if bbox_width < 10 or bbox_height < 10:
        return None

    cut_width = max(4, int(bbox_width * random.uniform(0.20, 0.45)))
    cut_height = max(4, int(bbox_height * random.uniform(0.20, 0.45)))

    bbox_area = bbox_width * bbox_height
    cut_area = cut_width * cut_height

    if cut_area > bbox_area * MAX_OCCLUSION_RATIO:
        # shrink the cutout patch so it never covers more than the occlusion cap
        scale = np.sqrt((bbox_area * MAX_OCCLUSION_RATIO) / cut_area)
        cut_width = max(4, int(cut_width * scale))
        cut_height = max(4, int(cut_height * scale))

    max_x = xmax - cut_width
    max_y = ymax - cut_height

    if max_x <= xmin or max_y <= ymin:
        return None

    cut_xmin = random.randint(xmin, max_x)
    cut_ymin = random.randint(ymin, max_y)
    cut_xmax = cut_xmin + cut_width
    cut_ymax = cut_ymin + cut_height

    target_region = image[ymin:ymax, xmin:xmax]

    if target_region.size == 0:
        return None

    # median color blends into the surrounding patch instead of a jarring solid block
    fill_color = np.median(target_region.reshape(-1, 3), axis=0).astype(np.uint8)

    result = image.copy()
    result[cut_ymin:cut_ymax, cut_xmin:cut_xmax] = fill_color

    return result


def choose_target_object(objects):
    hard_objects = [
        obj for obj in objects
        if CLASS_NAMES[obj["class_id"]] in HARD_CLASSES
    ]
    return random.choice(hard_objects if hard_objects else objects)


def get_cutout_probability(objects):
    image_classes = {CLASS_NAMES[obj["class_id"]] for obj in objects}
    return (
        HARD_CLASS_CUTOUT_PROB
        if image_classes & HARD_CLASSES
        else NORMAL_CUTOUT_PROB
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if V2_ROOT.exists():
        shutil.rmtree(V2_ROOT)

    shutil.copytree(V1_ROOT, V2_ROOT)

    # copytree carries over v1's data.yaml verbatim, whose `path:` still
    # points at V1_ROOT -- rewrite it to V2_ROOT so the copy is self-consistent.
    v2_yaml = V2_ROOT / "data.yaml"
    v2_yaml.write_text(
        v2_yaml.read_text(encoding="utf-8").replace(str(V1_ROOT), str(V2_ROOT)),
        encoding="utf-8",
    )

    source_images = V1_ROOT / "images" / "train"
    source_labels = V1_ROOT / "labels" / "train"
    output_images = V2_ROOT / "images" / "train"
    output_labels = V2_ROOT / "labels" / "train"

    images = sorted(
        path for path in source_images.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and "_aug_" not in path.stem  # skip v1's photometric augmentations
            and "_cutout_" not in path.stem  # skip already-cutout images (idempotency)
        )
    )

    stats = Counter()

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        objects = read_yolo_labels(label_path)

        if not objects or random.random() >= get_cutout_probability(objects):
            continue

        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]

        target = choose_target_object(objects)
        bbox = yolo_to_xyxy(target, image_width, image_height)
        cutout_image = apply_bbox_cutout(image, bbox)

        if cutout_image is None:
            continue

        new_stem = f"{image_path.stem}_cutout_v2"

        cv2.imwrite(
            str(output_images / f"{new_stem}{image_path.suffix}"),
            cutout_image,
        )
        shutil.copy2(label_path, output_labels / f"{new_stem}.txt")

        target_class = CLASS_NAMES[target["class_id"]]
        stats["total"] += 1
        stats[target_class] += 1

    report = [
        "V2 BBox-aware CutOut Augmentation Report",
        f"New CutOut images: {stats['total']}",
        "",
        "Target class distribution:",
    ]

    for class_name in CLASS_NAMES:
        if stats[class_name] > 0:
            report.append(f"{class_name}: {stats[class_name]}")

    (V2_ROOT / "augmentation_v2_report.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"CutOut images created: {stats['total']}")
    print(f"Output: {V2_ROOT}")


if __name__ == "__main__":
    main()
