"""Split the converted YOLO dataset into train/val/test (70/20/10, seeded)."""
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, IMAGE_SUFFIXES, SEED, YOLO_ROOT


def main():
    random.seed(SEED)

    source_images = YOLO_ROOT / "images" / "all"
    source_labels = YOLO_ROOT / "labels" / "all"

    images = sorted(
        p for p in source_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    class_groups = defaultdict(list)

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        class_ids = [
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Group by the lowest class id present so every class gets a
        # proportional 70/20/10 split, instead of splitting the dataset as
        # one undifferentiated pool.
        main_class = min(class_ids) if class_ids else -1
        class_groups[main_class].append(image_path)

    splits = {"train": [], "val": [], "test": []}

    for group in class_groups.values():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * 0.70)
        n_val = int(n * 0.20)

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    for split, split_images in splits.items():
        out_images = YOLO_ROOT / "images" / split
        out_labels = YOLO_ROOT / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, out_images / image_path.name)
            label_path = source_labels / f"{image_path.stem}.txt"
            shutil.copy2(label_path, out_labels / label_path.name)

        print(f"{split}: {len(split_images)} images")

    data_yaml = f"""path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        data_yaml += f"  {i}: {name}\n"

    (YOLO_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


if __name__ == "__main__":
    main()
