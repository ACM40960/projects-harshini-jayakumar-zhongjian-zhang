"""Verify the integrity of the final (v1) YOLO dataset and report results."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENHANCED_ROOT, CLASS_NAMES, IMAGE_SUFFIXES


def main() -> None:
    results = {}
    passed = True

    for split in ["train", "val", "test"]:
        image_dir = ENHANCED_ROOT / "images" / split
        label_dir = ENHANCED_ROOT / "labels" / split

        images = {
            p.stem for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {p.stem for p in label_dir.glob("*.txt")}

        missing = images - labels  # image with no matching label file
        orphan = labels - images  # label file with no matching image
        empty = 0
        invalid = 0
        objects = 0
        class_count = Counter()

        for label_path in label_dir.glob("*.txt"):
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not lines:
                empty += 1
                continue

            file_invalid = False

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    file_invalid = True
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    file_invalid = True
                    continue

                if not 0 <= class_id < len(CLASS_NAMES):
                    file_invalid = True
                if not all(0 <= value <= 1 for value in (x, y, w, h)):
                    file_invalid = True
                if w <= 0 or h <= 0:
                    file_invalid = True

                objects += 1
                class_count[class_id] += 1

            invalid += int(file_invalid)

        results[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "missing": len(missing),
            "orphan": len(orphan),
            "empty": empty,
            "invalid": invalid,
            "classes": class_count,
        }

        if missing or orphan or empty or invalid:
            passed = False

    report_lines = [
        "YOLO Dataset Final Verification Report",
        "=" * 70,
        "",
        f"Total Classes: {len(CLASS_NAMES)}",
        f"Total Objects: {sum(r['objects'] for r in results.values())}",
        "",
    ]

    for split, r in results.items():
        report_lines += [
            split.upper(),
            "-" * 70,
            f"Images: {r['images']}",
            f"Labels: {r['labels']}",
            f"Objects: {r['objects']}",
            f"Missing Labels: {r['missing']}",
            f"Orphan Labels: {r['orphan']}",
            f"Empty Labels: {r['empty']}",
            f"Invalid Labels: {r['invalid']}",
            "",
        ]

    report_lines += [
        "Class Distribution",
        "-" * 70,
        f"{'Class':<20}{'Train':>10}{'Val':>10}{'Test':>10}{'Total':>10}",
    ]

    for class_id, name in enumerate(CLASS_NAMES):
        train = results["train"]["classes"][class_id]
        val = results["val"]["classes"][class_id]
        test = results["test"]["classes"][class_id]
        report_lines.append(
            f"{name:<20}{train:>10}{val:>10}{test:>10}{train + val + test:>10}"
        )

    report_lines += [
        "",
        "Final Conclusion",
        "-" * 70,
        "Dataset verification PASSED."
        if passed else
        "Dataset verification FAILED.",
    ]

    report = "\n".join(report_lines)
    print(report)

    report_path = ENHANCED_ROOT / "final_dataset_verification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
