"""
Count vehicles per image for a folder laid out as:

    sample/thon/<camera_id>/<image>.jpg
    sample/phranakhon/<camera_id>/<image>.jpg

Run from the repo root (the folder that contains Models/):

    # images already on disk
    python count_sample.py --input-dir sample --output-csv sample_counts.csv

    # images on YOUR Google Drive (share as "Anyone with the link"):
    python count_sample.py --gdrive-zip <FILE_ID_OF_sample.zip> --input-dir sample
    python count_sample.py --gdrive-folder "https://drive.google.com/drive/folders/<ID>" --input-dir sample
    python count_sample.py --input-dir sample --limit 10          # quick test
    python count_sample.py --input-dir sample --save-annotated out_annotated

Output: one CSV row per image with district, camera_id, file name,
timestamp (if the file name follows <cam>_<desc>_YYYYMMDD_HHMMSS.jpg),
and counts for car / motorcycle / bus / truck / total_vehicles.
"""
import argparse
import csv
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Models.detector import YOLOVehicleDetector, imwrite_unicode  # noqa: E402
from Models.model_registry import resolve_model  # noqa: E402
from Models.pipeline import parse_cctv_filename  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
FIELDS = ["district", "camera_id", "image_file", "timestamp", "date", "hour", "minute",
          "car", "motorcycle", "bus", "truck", "total_vehicles", "relative_path"]


def fetch_from_drive(args, root: Path):
    """Download the images from the person's own Google Drive into root (needs: pip install gdown)."""
    import gdown
    root.mkdir(parents=True, exist_ok=True)
    if args.gdrive_zip:
        zip_path = root.parent / "_drive_images.zip"
        gdown.download(id=args.gdrive_zip, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        zip_path.unlink(missing_ok=True)
    elif args.gdrive_folder:
        # NOTE: gdown only lists the first 50 files per Drive folder -> prefer --gdrive-zip for big sets
        gdown.download_folder(url=args.gdrive_folder, output=str(root), quiet=False, use_cookies=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="sample")
    ap.add_argument("--output-csv", default="sample_counts.csv")
    ap.add_argument("--model", default=None, help="Default: Models/weights/best.pt")
    ap.add_argument("--conf", type=float, default=None,
                    help="Fixed confidence. Default: adaptive (0.28 day / 0.20 night, from file-name time).")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'. Default: auto.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gdrive-zip", default=None, help="Drive file ID of a zip of the sample/ folder")
    ap.add_argument("--gdrive-folder", default=None, help="Drive folder URL (max 50 files per folder)")
    ap.add_argument("--save-annotated", default="", help="Folder to save images with boxes (optional)")
    args = ap.parse_args()

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    root = Path(args.input_dir).resolve()
    if args.gdrive_zip or args.gdrive_folder:
        fetch_from_drive(args, root)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if args.limit:
        images = images[:args.limit]
    print(f"[*] {len(images)} images under {root} | device={device}")

    detector = YOLOVehicleDetector(model_name=resolve_model(args.model), device=device)
    ann_dir = Path(args.save_annotated) if args.save_annotated else None
    if ann_dir:
        ann_dir.mkdir(parents=True, exist_ok=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, p in enumerate(images, 1):
            rel = p.relative_to(root)
            # .../<district>/<camera_id>/<file> -> take the last three parts, so an extra
            # top folder (e.g. a "sample/" inside the zip) does not matter
            district = rel.parts[-3] if len(rel.parts) >= 3 else ""
            camera_id = rel.parts[-2] if len(rel.parts) >= 2 else ""
            parsed = parse_cctv_filename(p.name) or {}

            res = detector.predict(p, conf=args.conf, draw_annotations=bool(ann_dir))
            c = res["counts"]
            w.writerow({
                "district": district,
                "camera_id": camera_id,
                "image_file": p.name,
                "timestamp": parsed.get("timestamp", ""),
                "date": parsed.get("date", ""),
                "hour": parsed.get("hour", ""),
                "minute": parsed.get("minute", ""),
                "car": c.get("car", 0),
                "motorcycle": c.get("motorcycle", 0),
                "bus": c.get("bus", 0),
                "truck": c.get("truck", 0),
                "total_vehicles": c.get("total_vehicles", 0),
                "relative_path": rel.as_posix(),
            })
            if ann_dir and res["annotated_image"] is not None:
                imwrite_unicode(ann_dir / f"{district}_{camera_id}_{p.name}", res["annotated_image"])
            if i % 50 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] {rel.as_posix()} -> total {c.get('total_vehicles', 0)}")

    print(f"[+] Saved {out}")


if __name__ == "__main__":
    main()
