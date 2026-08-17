"""Report per-class object/image counts for each dataset split."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import YOLO_ROOT, CLASS_NAMES


def main() -> None:
    for split in ["train", "val", "test"]:
        label_dir = YOLO_ROOT / "labels" / split
        class_count = Counter()
        image_count = Counter()

        for label_path in label_dir.glob("*.txt"):
            ids = [
                int(line.split()[0])
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            class_count.update(ids)
            image_count.update(set(ids))  # de-duplicated: images containing the class, not object count

        print(f"\n{split.upper()}")
        print(f"Images: {len(list(label_dir.glob('*.txt')))}")
        print(f"Objects: {sum(class_count.values())}")

        for class_id, name in enumerate(CLASS_NAMES):
            print(
                f"{name:<15} "
                f"objects={class_count[class_id]:<5} "
                f"images={image_count[class_id]}"
            )


if __name__ == "__main__":
    main()
