"""Shared paths and constants for the dataset preprocessing pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"

VOC_IMAGES = ROOT / "raw-dataset" / "JPEGImages"
VOC_XML = ROOT / "raw-dataset" / "Annotations"

YOLO_ROOT = ROOT / "yolo_dataset"
ENHANCED_ROOT = ROOT / "enhanced_yolo_dataset"
ENHANCED_ROOT_V2 = ROOT / "enhanced_yolo_dataset_v2"

CLASS_NAMES = [
    "RaccoonDog",
    "Hare",
    "MuskDeer",
    "LeopardCat",
    "RedFox",
    "WildBoar",
    "SikaDeer",
    "RoeDeer",
    "AmurTiger",
    "Weasel",
    "Leopard",
    "Sable",
    "BlackBear",
    "Badger",
    "Y.T.Marten",
    "Dog",
    "Cow",
]

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

SEED = 42
