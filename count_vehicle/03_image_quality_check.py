from collections import defaultdict
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat, UnidentifiedImageError

from src.config import RAW_DIR, OUTPUT_DIR

IMAGE_RAW_DIR = RAW_DIR / "cctv_images"

CANDIDATES_FILE = OUTPUT_DIR / "camera_candidates.csv"

# Output folders. Original images in data/raw will NOT be changed.
QUALITY_OUTPUT_DIR = OUTPUT_DIR / "image_quality"
VALID_DIR = QUALITY_OUTPUT_DIR / "valid"
INVALID_DIR = QUALITY_OUTPUT_DIR / "invalid"
REPORT_FILE = QUALITY_OUTPUT_DIR / "image_quality_report.csv"

# Only process images captured during these dates (inclusive).
START_DATE = "20260825"
END_DATE = "20260831"

# Image extensions to scan.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Conservative blank-image thresholds.
# Tune these after manually reviewing a sample of valid/invalid images.
WHITE_MEAN_THRESHOLD = 245
WHITE_STD_THRESHOLD = 15
WHITE_PIXEL_THRESHOLD = 0.98

BLACK_MEAN_THRESHOLD = 8
BLACK_STD_THRESHOLD = 12

UNIFORM_STD_THRESHOLD = 2.5

# Compare each frame with the usual row transitions and noise for its camera.
GLITCH_ROW_EXCESS_THRESHOLD = 40.0
GLITCH_NOISE_EXCESS_THRESHOLD = 15.0
DARK_MEAN_THRESHOLD = 60.0
ROW_PROFILE_BINS = 90

# Resize image before calculating statistics to make processing faster.
ANALYSIS_SIZE = (512, 512)


# =========================================================
# 2. Helper functions
# =========================================================

def normalize_camera_id(value):
    """Normalize camera IDs so values such as 001 and 1 are handled consistently."""
    value = str(value).strip()

    # pandas may read an ID like 123 as 123.0 in some CSV files.
    if value.endswith(".0"):
        value = value[:-2]

    return value


def parse_filename(filename):
    """
    Expected filename format:
    cameraid_location_locationdetail_YYYYMMDD_HHMMSS.jpg

    Parse from the right so location/location_detail may contain underscores.
    camera_id is taken from the first segment.
    """
    pattern = (
        r"^(?P<camera_id>[^_]+)_.+_"
        r"(?P<date>\d{8})_(?P<time>\d{6})"
        r"\.(?P<extension>jpg|jpeg|png)$"
    )

    match = re.match(pattern, filename, flags=re.IGNORECASE)

    if not match:
        return None

    return {
        "camera_id": normalize_camera_id(match.group("camera_id")),
        "date": match.group("date"),
        "time": match.group("time"),
    }


def inspect_image(image_path):
    """
    Check whether an image can be decoded and whether it looks like
    a blank white, blank black, or almost completely uniform image.

    Returns:
        is_valid, reason, width, height, mean_brightness, brightness_std,
        white_pixel_ratio, row_profile, noise_level
    """
    try:
        # First verify that the file is a readable image.
        with Image.open(image_path) as image:
            image.verify()

        # Reopen after verify() to inspect pixels.
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            if width == 0 or height == 0:
                return (
                    False, "invalid_dimensions", width, height,
                    None, None, None, None, None
                )

            image.thumbnail(ANALYSIS_SIZE)

            # Convert to grayscale for brightness analysis.
            grayscale = image.convert("L")
            stats = ImageStat.Stat(grayscale)

            mean_brightness = float(stats.mean[0])
            brightness_std = float(stats.stddev[0])
            gray_array = np.asarray(grayscale, dtype=np.float32)
            row_diff = np.abs(np.diff(gray_array, axis=0)).mean(axis=1)
            row_profile = np.array(
                [
                    float(segment.max()) if segment.size else 0.0
                    for segment in np.array_split(row_diff, ROW_PROFILE_BINS)
                ],
                dtype=np.float32,
            )
            noise_level = float(
                np.median(np.abs(np.diff(gray_array, axis=1)))
            )

            histogram = grayscale.histogram()
            total_pixels = sum(histogram)

            white_pixels = sum(histogram[250:256])
            white_pixel_ratio = (
                white_pixels / total_pixels if total_pixels else 0.0
            )

            # Fully/near-white frame.
            if (
                mean_brightness >= WHITE_MEAN_THRESHOLD
                and brightness_std <= WHITE_STD_THRESHOLD
            ) or white_pixel_ratio >= WHITE_PIXEL_THRESHOLD:
                return (
                    False,
                    "blank_or_overexposed_white",
                    width,
                    height,
                    mean_brightness,
                    brightness_std,
                    white_pixel_ratio,
                    row_profile,
                    noise_level,
                )

            # Fully/near-black frame.
            if (
                mean_brightness <= BLACK_MEAN_THRESHOLD
                and brightness_std <= BLACK_STD_THRESHOLD
            ):
                return (
                    False,
                    "blank_or_black",
                    width,
                    height,
                    mean_brightness,
                    brightness_std,
                    white_pixel_ratio,
                    row_profile,
                    noise_level,
                )

            # Almost no variation across the whole image.
            if brightness_std <= UNIFORM_STD_THRESHOLD:
                return (
                    False,
                    "almost_uniform_image",
                    width,
                    height,
                    mean_brightness,
                    brightness_std,
                    white_pixel_ratio,
                    row_profile,
                    noise_level,
                )

            return (
                True,
                "valid",
                width,
                height,
                mean_brightness,
                brightness_std,
                white_pixel_ratio,
                row_profile,
                noise_level,
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        return (
            False,
            f"unreadable_image: {type(error).__name__}",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flag_glitched_images(records):
    """Flag row/noise anomalies relative to each camera's day/night baseline."""
    profiles_by_camera = defaultdict(list)

    for record in records:
        if record["status"] != "valid" or record["row_profile"] is None:
            continue

        is_dark = record["mean_brightness"] < DARK_MEAN_THRESHOLD
        profiles_by_camera[(record["camera_id"], is_dark)].append(record)

    references = {}
    for key, camera_records in profiles_by_camera.items():
        references[key] = (
            np.median(
                np.stack([record["row_profile"] for record in camera_records]),
                axis=0,
            ),
            float(np.median([record["noise_level"] for record in camera_records])),
        )

    for record in records:
        record["row_excess_max"] = None
        record["noise_excess"] = None

        if record["status"] != "valid" or record["row_profile"] is None:
            continue

        is_dark = record["mean_brightness"] < DARK_MEAN_THRESHOLD
        reference = references.get((record["camera_id"], is_dark))
        if reference is None:
            continue

        baseline_profile, baseline_noise = reference
        row_excess_max = float(
            np.max(record["row_profile"] - baseline_profile)
        )
        noise_excess = float(record["noise_level"] - baseline_noise)
        record["row_excess_max"] = row_excess_max
        record["noise_excess"] = noise_excess

        if (
            row_excess_max > GLITCH_ROW_EXCESS_THRESHOLD
            or noise_excess > GLITCH_NOISE_EXCESS_THRESHOLD
        ):
            record["status"] = "invalid"
            record["reason"] = "glitch"

    return records


def copy_to_output(source_path, destination_root):
    """
    Copy image to output while preserving its relative folder structure.
    This prevents files with identical names in different subfolders
    from overwriting one another.
    """
    relative_path = source_path.relative_to(IMAGE_RAW_DIR)
    destination_path = destination_root / relative_path
    other_root = INVALID_DIR if destination_root == VALID_DIR else VALID_DIR
    old_destination = other_root / relative_path

    if old_destination.exists():
        old_destination.unlink()

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)

    return destination_path


# =========================================================
# 3. Main image screening process
# =========================================================

def main():
    print("Starting CCTV image quality check...")

    # Check input paths.
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ camera candidates: {CANDIDATES_FILE}\n"
            "กรุณารัน python -m src.02_camera_sampling ก่อน"
        )

    if not IMAGE_RAW_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์รูปภาพ: {IMAGE_RAW_DIR}\n"
            "กรุณาตรวจสอบว่าโฟลเดอร์รูป raw อยู่ที่ data/raw/images/"
        )

    # Load selected camera IDs.
    candidates_df = pd.read_csv(
        CANDIDATES_FILE,
        dtype={"camera_id": str},
    )

    if "camera_id" not in candidates_df.columns:
        raise ValueError(
            "camera_candidates.csv ต้องมีคอลัมน์ camera_id"
        )

    candidate_ids = {
        normalize_camera_id(camera_id)
        for camera_id in candidates_df["camera_id"].dropna()
    }

    print(f"Candidate cameras: {len(candidate_ids)}")
    print(f"Image raw folder: {IMAGE_RAW_DIR}")
    print(f"Date range: {START_DATE} - {END_DATE}")

    # Create output folders.
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    INVALID_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find image files recursively.
    all_image_paths = [
        path
        for path in IMAGE_RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nImage files found: {len(all_image_paths)}")

    records = []
    skipped_non_candidate = 0
    skipped_outside_date = 0
    skipped_bad_filename = 0

    for index, image_path in enumerate(all_image_paths, start=1):
        parsed = parse_filename(image_path.name)

        if parsed is None:
            skipped_bad_filename += 1
            continue

        camera_id = parsed["camera_id"]
        capture_date = parsed["date"]
        capture_time = parsed["time"]

        # Keep only images from selected candidate cameras.
        if camera_id not in candidate_ids:
            skipped_non_candidate += 1
            continue

        # Keep only the requested date range.
        if not (START_DATE <= capture_date <= END_DATE):
            skipped_outside_date += 1
            continue

        (
            is_valid,
            reason,
            width,
            height,
            mean_brightness,
            brightness_std,
            white_pixel_ratio,
            row_profile,
            noise_level,
        ) = inspect_image(image_path)

        records.append({
            "camera_id": camera_id,
            "date": capture_date,
            "time": capture_time,
            "filename": image_path.name,
            "source_path": str(image_path),
            "output_path": None,
            "status": "valid" if is_valid else "invalid",
            "reason": reason,
            "width": width,
            "height": height,
            "mean_brightness": mean_brightness,
            "brightness_std": brightness_std,
            "white_pixel_ratio": white_pixel_ratio,
            "row_profile": row_profile,
            "noise_level": noise_level,
            "row_excess_max": None,
            "noise_excess": None,
        })

        if index % 500 == 0:
            print(f"Scanned {index}/{len(all_image_paths)} files...")

    flag_glitched_images(records)

    for record in records:
        destination_root = (
            VALID_DIR if record["status"] == "valid" else INVALID_DIR
        )
        copied_path = copy_to_output(
            Path(record["source_path"]),
            destination_root
        )
        record["output_path"] = str(copied_path)

    # Save report.
    report_columns = [
        "camera_id",
        "date",
        "time",
        "filename",
        "source_path",
        "output_path",
        "status",
        "reason",
        "width",
        "height",
        "mean_brightness",
        "brightness_std",
        "white_pixel_ratio",
        "row_excess_max",
        "noise_excess",
    ]

    report_df = pd.DataFrame(records, columns=report_columns)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    # Summary.
    valid_count = (
        int((report_df["status"] == "valid").sum())
        if not report_df.empty else 0
    )
    invalid_count = (
        int((report_df["status"] == "invalid").sum())
        if not report_df.empty else 0
    )
    glitch_count = (
        int((report_df["reason"] == "glitch").sum())
        if not report_df.empty else 0
    )

    print("\n========== IMAGE QUALITY SUMMARY ==========")
    print(f"Candidate cameras       : {len(candidate_ids)}")
    print(f"Images found            : {len(all_image_paths)}")
    print(f"Images processed        : {len(report_df)}")
    print(f"Valid images            : {valid_count}")
    print(f"Invalid images          : {invalid_count}")
    print(f"Glitch images           : {glitch_count}")
    print(f"Skipped non-candidates  : {skipped_non_candidate}")
    print(f"Skipped outside dates   : {skipped_outside_date}")
    print(f"Skipped bad filenames   : {skipped_bad_filename}")

    if not report_df.empty:
        print("\nInvalid reason summary:")
        invalid_reasons = report_df.loc[
            report_df["status"] == "invalid", "reason"
        ].value_counts()
        print(invalid_reasons if not invalid_reasons.empty else "No invalid images.")

    print("\nOutput folders:")
    print(f"Valid images   : {VALID_DIR}")
    print(f"Invalid images : {INVALID_DIR}")
    print(f"CSV report     : {REPORT_FILE}")

    print("\nImage quality check completed.")


if __name__ == "__main__":
    main()
