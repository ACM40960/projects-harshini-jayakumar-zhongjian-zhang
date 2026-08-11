"""Analyze the raw Pascal VOC dataset and report image/annotation statistics."""
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VOC_IMAGES, VOC_XML, IMAGE_SUFFIXES


def main():
    images = {
        p.stem: p for p in VOC_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    xmls = {p.stem: p for p in VOC_XML.glob("*.xml")}

    class_count = Counter()
    size_count = Counter()
    invalid_boxes = 0
    empty_xml = 0
    broken_images = 0
    objects = 0

    for image_path in images.values():
        try:
            with Image.open(image_path) as im:
                size_count[im.size] += 1
        except Exception:
            broken_images += 1

    for xml_path in xmls.values():
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        xml_objects = root.findall("object")

        if not xml_objects:
            empty_xml += 1

        for obj in xml_objects:
            name = obj.findtext("name").strip()
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            class_count[name] += 1
            objects += 1

            # A box is invalid if it's out-of-bounds for the image or has zero/negative area.
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                invalid_boxes += 1

    print(f"Images: {len(images)}")
    print(f"XML files: {len(xmls)}")
    print(f"Objects: {objects}")
    print(f"Invalid boxes: {invalid_boxes}")
    print(f"Broken images: {broken_images}")
    print(f"XML without image: {len(set(xmls) - set(images))}")
    print(f"Image without XML: {len(set(images) - set(xmls))}")
    print(f"Empty annotations: {empty_xml}")

    print("\nImage sizes:")
    for size, count in size_count.most_common():
        print(f"{size[0]}x{size[1]}: {count}")

    print("\nClass distribution:")
    for name, count in class_count.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
