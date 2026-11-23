import random
import pandas as pd

from pyproj import Transformer

from src.config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE
)


# ==========================================
# 1. File paths
# ==========================================

INPUT_FILE = PROCESSED_DIR / "camera_classification.csv"

CANDIDATES_FILE = OUTPUT_DIR / "camera_candidates.csv"

SELECTED_FILE = OUTPUT_DIR / "selected_cameras.csv"

SUMMARY_FILE = OUTPUT_DIR / "sampling_summary.csv"


# ==========================================
# 2. Coordinate transformation
# ==========================================

# WGS84 -> UTM Zone 47N
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32647",
    always_xy=True
)


def convert_coordinates(df):

    x, y = transformer.transform(
        df["longitude"].tolist(),
        df["latitude"].tolist()
    )

    result = df.copy()

    result["x"] = x
    result["y"] = y

    return result


# ==========================================
# 3. Spatially balanced sampling
# ==========================================

def spatial_sampling(candidates, sample_size):

    rng = random.Random(RANDOM_SEED)

    candidates = convert_coordinates(candidates)

    selected = []

    selected_counts = {
        road_type: 0
        for road_type in sample_size
    }

    total_required = sum(sample_size.values())

    while len(selected) < total_required:

        # ประเภทถนนที่ยังเลือกไม่ครบ
        available_types = [
            road_type
            for road_type, required in sample_size.items()
            if selected_counts[road_type] < required
        ]

        # Candidate ที่ยังไม่ได้เลือก
        selected_ids = {
            row["camera_id"]
            for row in selected
        }

        available_candidates = candidates[
            candidates["road_type"].isin(available_types)
            & ~candidates["camera_id"].isin(selected_ids)
        ].copy()

        if available_candidates.empty:
            raise ValueError(
                "ไม่สามารถเลือกกล้องได้ครบตามจำนวนที่ต้องการ"
            )

        # ถ้ายังไม่มีตัวแรก ให้สุ่มเลือกหนึ่งตัว
        if not selected:

            first_index = rng.choice(
                available_candidates.index.tolist()
            )

            chosen = available_candidates.loc[first_index]

        else:

            # พิกัดกล้องที่เลือกไปแล้ว
            selected_coordinates = [
                (row["x"], row["y"])
                for row in selected
            ]

            # คำนวณระยะห่างจากกล้องที่เลือกไว้ทั้งหมด
            min_distances = []

            for _, candidate in available_candidates.iterrows():

                distances = [
                    (
                        (candidate["x"] - x) ** 2
                        + (candidate["y"] - y) ** 2
                    ) ** 0.5
                    for x, y in selected_coordinates
                ]

                # ระยะจากกล้องที่ใกล้ที่สุด
                min_distances.append(
                    min(distances)
                )

            available_candidates["min_distance"] = min_distances

            # เลือกตัวที่ห่างจากกล้องเดิมมากที่สุด
            max_distance = available_candidates[
                "min_distance"
            ].max()

            best_candidates = available_candidates[
                available_candidates["min_distance"] == max_distance
            ]

            chosen_index = rng.choice(
                best_candidates.index.tolist()
            )

            chosen = available_candidates.loc[chosen_index]

        selected.append(chosen.to_dict())

        selected_counts[chosen["road_type"]] += 1

        print(
            f"Selected {len(selected)}/{total_required}: "
            f"Camera {chosen['camera_id']} | "
            f"{chosen['road_type']}"
        )

    result = pd.DataFrame(selected)

    # เอาคอลัมน์ช่วยคำนวณออก
    result = result.drop(
        columns=["x", "y", "min_distance"],
        errors="ignore"
    )

    return result


# ==========================================
# 4. Main
# ==========================================

def main():

    print("Reading camera classification...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "ไม่พบ camera_classification.csv "
            "กรุณารัน 01_create_template.py ก่อน"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "camera_id",
        "latitude",
        "longitude",
        "side",
        "road_type"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"ไม่พบคอลัมน์ {column}"
            )

    # เลือกกล้องฝั่งพระนคร
    candidates = df[
        (df["side"] == "PhraNakhon")
        & df["road_type"].isin(SAMPLE_SIZE)
    ].copy()

    if candidates["camera_id"].duplicated().any():
        raise ValueError(
            "พบ camera_id ซ้ำใน candidate"
        )

    # ตรวจจำนวน candidate
    print("\nCandidate summary:")

    print(
        candidates["road_type"].value_counts()
    )

    # ตรวจจำนวนกล้องแต่ละประเภทว่าพอหรือไม่
    available_sample_size = {}

    for road_type, required_count in SAMPLE_SIZE.items():

        available_count = len(
            candidates[
                candidates["road_type"] == road_type
            ]
        )

        available_sample_size[road_type] = min(
            required_count,
            available_count
        )

        if available_count < required_count:
            print(
                f"Reducing {road_type} sample from {required_count} "
                f"to {available_count}: not enough classifiable cameras."
            )

    if sum(available_sample_size.values()) == 0:
        raise ValueError(
            "ไม่พบกล้องฝั่งพระนครที่จำแนกประเภทถนนได้สำหรับสุ่มเลือก"
        )

    # บันทึก Candidate
    candidates.to_csv(
        CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Spatial sampling
    print("\nStarting spatially balanced sampling...")

    selected = spatial_sampling(
        candidates,
        available_sample_size
    )

    # เรียงลำดับผลลัพธ์
    selected = selected.sort_values(
        by=["road_type", "camera_id"]
    ).reset_index(drop=True)

    # บันทึกกล้องที่เลือก
    selected.to_csv(
        SELECTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # สรุปผล
    summary = (
        selected
        .groupby(["side", "road_type"])
        .size()
        .reset_index(name="camera_count")
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSampling completed!")

    print(f"\nCandidates: {CANDIDATES_FILE}")
    print(f"Selected: {SELECTED_FILE}")
    print(f"Summary: {SUMMARY_FILE}")

    print("\nSelected cameras:")

    print(
        selected[
            ["camera_id", "location", "side", "road_type"]
        ].to_string(index=False)
    )

    print("\nSampling summary:")

    print(summary.to_string(index=False))

    print(f"\nTotal selected: {len(selected)} cameras")


if __name__ == "__main__":
    main()