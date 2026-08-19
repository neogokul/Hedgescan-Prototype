"""
Rename a folder of hedgerow photos to sequential numbers (1.jpg, 2.jpg, ...)
ready to commit into the data/ folder for labeling.

Usage:
    python rename_dataset_images.py <source_folder> [destination_folder]

Images are sorted by filename before renumbering; only .jpg/.jpeg/.png files
are copied, and each keeps its original extension.
"""

import shutil
import sys
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def rename_dataset(source_dir: Path, dest_dir: Path) -> int:
    images = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(images, start=1):
        shutil.copy2(src, dest_dir / f"{index}{src.suffix.lower()}")
    return len(images)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <source_folder> [destination_folder]")
        sys.exit(1)

    source = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    count = rename_dataset(source, dest)
    print(f"Copied and renamed {count} images from {source} to {dest}")
