"""Giải nén dữ liệu BTC từ thư mục ZIP/ vào data/.

Tự động xử lý mapping tên thư mục gốc trong zip sang thư mục đích mà code đọc:
  - clip-features-32/ → data/clip-features/
  - keyframes/        → data/keyframes/
  - map-keyframes/    → data/map-keyframes/
  - media-info/       → data/media-info/
  - objects/          → data/objects/
"""
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ZIP_DIR = ROOT_DIR / "ZIP"
DATA_DIR = ROOT_DIR / "data"

# Mapping: prefix bên trong zip → thư mục đích trong data/
PREFIX_MAP = {
    "clip-features-32": "clip-features",
    "clip-features-16": "clip-features",
    "clip-features":    "clip-features",
    "keyframes":        "keyframes",
    "map-keyframes":    "map-keyframes",
    "media-info":       "media-info",
    "objects":          "objects",
}


def _detect_prefix(zip_path: Path) -> tuple[str, str]:
    """Detect root prefix inside zip and map to target directory."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and parts[0]:
                root_prefix = parts[0]
                for known_prefix, target in PREFIX_MAP.items():
                    if root_prefix == known_prefix or root_prefix.startswith(known_prefix):
                        return root_prefix, target
                # Fallback: dùng tên zip file để đoán
                stem = zip_path.stem.lower()
                for known_prefix, target in PREFIX_MAP.items():
                    if known_prefix in stem:
                        return root_prefix, target
                return root_prefix, root_prefix
    return "", ""


def extract_zip(zip_path: Path, dry_run: bool = False) -> int:
    """Extract one zip file to the correct data/ subdirectory."""
    root_prefix, target_dir_name = _detect_prefix(zip_path)
    if not root_prefix:
        print(f"  SKIP: Could not detect prefix in {zip_path.name}")
        return 0

    target_dir = DATA_DIR / target_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {zip_path.name}")
    print(f"    zip prefix: '{root_prefix}/' -> target: data/{target_dir_name}/")

    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Skip directory entries
            if member.endswith("/"):
                continue

            # Strip root prefix: "clip-features-32/L21_V001.npy" → "L21_V001.npy"
            if member.startswith(root_prefix + "/"):
                relative = member[len(root_prefix) + 1:]
            else:
                relative = member

            if not relative:
                continue

            dest = target_dir / relative
            if dest.exists():
                continue  # Skip already extracted

            if dry_run:
                print(f"    [dry-run] {relative}")
                extracted += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"    Extracted {extracted} files" + (" (dry-run)" if dry_run else ""))
    return extracted


def main():
    dry_run = "--dry-run" in sys.argv

    if not ZIP_DIR.exists():
        print(f"ERROR: ZIP directory not found: {ZIP_DIR}")
        return

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in {ZIP_DIR}")
        return

    print(f"Found {len(zip_files)} zip files in {ZIP_DIR}")
    if dry_run:
        print("(DRY RUN - no files will be extracted)\n")
    print()

    total = 0
    for zp in zip_files:
        total += extract_zip(zp, dry_run=dry_run)

    print(f"\nTotal extracted: {total} files")
    if not dry_run:
        print("\nDone! Now run:")
        print("  python scripts/import_btc_data.py")


if __name__ == "__main__":
    main()
