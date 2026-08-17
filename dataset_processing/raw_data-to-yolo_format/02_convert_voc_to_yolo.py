"""Convert Pascal VOC XML annotations into YOLO-format label files."""
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, IMAGE_SUFFIXES, VOC_IMAGES, VOC_XML, YOLO_ROOT


def main() -> None:
    image_map = {
        p.stem: p for p in VOC_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    out_images = YOLO_ROOT / "images" / "all"
    out_labels = YOLO_ROOT / "labels" / "all"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    converted_objects = 0
    clipped_objects = 0

    for xml_path in sorted(VOC_XML.glob("*.xml")):
        image_path = image_map[xml_path.stem]
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        lines = []

        for obj in root.findall("object"):
            name = obj.findtext("name").strip()
            box = obj.find("bndbox")

            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            # Some VOC boxes slightly exceed the image bounds (annotation tool
            # rounding); clip instead of discarding so no object is lost.
            clipped = (xmin < 0 or ymin < 0 or xmax > width or ymax > height)
            xmin = max(0.0, min(xmin, width))
            ymin = max(0.0, min(ymin, height))
            xmax = max(0.0, min(xmax, width))
            ymax = max(0.0, min(ymax, height))

            # Clipping can collapse a box to zero area -- skip those.
            if xmax <= xmin or ymax <= ymin:
                continue

            # YOLO format: box center + size, normalized to [0, 1].
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            lines.append(
                f"{class_to_id[name]} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

            converted_objects += 1
            clipped_objects += int(clipped)

        shutil.copy2(image_path, out_images / image_path.name)
        (out_labels / f"{xml_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    (YOLO_ROOT / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )
    (YOLO_ROOT / "class_mapping.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8"
    )

    print(f"Converted images: {len(image_map)}")
    print(f"Converted objects: {converted_objects}")
    print(f"Clipped objects: {clipped_objects}")


if __name__ == "__main__":
    main()
