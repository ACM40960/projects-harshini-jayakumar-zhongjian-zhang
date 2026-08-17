"""Verify the integrity of the final (v2) YOLO dataset and report results."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, ENHANCED_ROOT_V2, IMAGE_SUFFIXES

DATASET_ROOT = ENHANCED_ROOT_V2


def verify_label(label_path: Path) -> tuple[int, Counter, bool, bool]:
    """Validate one YOLO label file's contents.

    Args:
        label_path: Path to a `.txt` label file (one object per line).

    Returns:
        A tuple of (object_count, class_count, is_empty, is_invalid).
    """
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        return 0, Counter(), True, True

    object_count = 0
    class_count = Counter()
    invalid = False

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            invalid = True
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            invalid = True
            continue

        if not 0 <= class_id < len(CLASS_NAMES):
            invalid = True
            continue

        if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
            invalid = True
            continue

        if w <= 0 or h <= 0:
            invalid = True
            continue

        object_count += 1
        class_count[class_id] += 1

    return object_count, class_count, False, invalid


def verify_split(split: str) -> dict:
    """Verify one dataset split (train/val/test) and summarize its contents.

    Args:
        split: Split name, one of "train", "val", "test".

    Returns:
        A dict of counts: images, labels, objects, missing_labels,
        orphan_labels, empty_labels, invalid_labels, class_count.
    """
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    images = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    }

    labels = {
        path.stem
        for path in label_dir.glob("*.txt")
    }

    missing_labels = images - labels  # image with no matching label file
    orphan_labels = labels - images  # label file with no matching image

    total_objects = 0
    empty_labels = 0
    invalid_labels = 0
    class_count = Counter()

    for label_path in label_dir.glob("*.txt"):
        objects, classes, is_empty, is_invalid = verify_label(label_path)

        total_objects += objects
        class_count.update(classes)
        empty_labels += int(is_empty)
        invalid_labels += int(is_invalid)

    return {
        "images": len(images),
        "labels": len(labels),
        "objects": total_objects,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "empty_labels": empty_labels,
        "invalid_labels": invalid_labels,
        "class_count": class_count,
    }


def main() -> None:
    splits = ["train", "val", "test"]
    results = {split: verify_split(split) for split in splits}

    passed = all(
        result["missing_labels"] == 0
        and result["orphan_labels"] == 0
        and result["empty_labels"] == 0
        and result["invalid_labels"] == 0
        for result in results.values()
    )

    total_images = sum(result["images"] for result in results.values())
    total_objects = sum(result["objects"] for result in results.values())

    report = [
        "V2 YOLO Dataset Verification Report",
        "=" * 70,
        f"Total Classes: {len(CLASS_NAMES)}",
        f"Total Images: {total_images}",
        f"Total Objects: {total_objects}",
        "",
    ]

    for split, result in results.items():
        report.extend([
            split.upper(),
            "-" * 70,
            f"Images: {result['images']}",
            f"Labels: {result['labels']}",
            f"Objects: {result['objects']}",
            f"Missing Labels: {result['missing_labels']}",
            f"Orphan Labels: {result['orphan_labels']}",
            f"Empty Labels: {result['empty_labels']}",
            f"Invalid Labels: {result['invalid_labels']}",
            "",
        ])

    report.extend([
        "Class Distribution",
        "-" * 70,
        f"{'Class':<20}{'Train':>10}{'Val':>10}{'Test':>10}{'Total':>10}",
    ])

    for class_id, class_name in enumerate(CLASS_NAMES):
        train = results["train"]["class_count"][class_id]
        val = results["val"]["class_count"][class_id]
        test = results["test"]["class_count"][class_id]

        report.append(
            f"{class_name:<20}"
            f"{train:>10}"
            f"{val:>10}"
            f"{test:>10}"
            f"{train + val + test:>10}"
        )

    report.extend([
        "",
        "Final Conclusion",
        "-" * 70,
        "Dataset verification PASSED."
        if passed
        else "Dataset verification FAILED.",
    ])

    report_text = "\n".join(report)
    print(report_text)

    output_file = DATASET_ROOT / "v2_dataset_verification_report.txt"
    output_file.write_text(report_text, encoding="utf-8")

    print(f"\nReport saved to:\n{output_file}")


if __name__ == "__main__":
    main()
