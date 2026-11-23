import time

import requests
import pandas as pd

from shapely.geometry import (
    shape,
    Point,
    LineString,
    box
)
from shapely.ops import unary_union, polygonize

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    OVERPASS_URL,
    BBOX_PADDING,
    PHRA_NAKHON_LAT,
    PHRA_NAKHON_LON
)


# ==========================================
# 1. File paths
# ==========================================

INPUT_FILE = RAW_DIR / "cameras.csv"

OUTPUT_FILE = PROCESSED_DIR / "camera_classification.csv"

OVERPASS_FALLBACK_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
)


# ==========================================
# 2. Road type classification
# ==========================================

INTERSECTION_KEYWORDS = [
    "แยก",
    "ทางแยก",
    "จุดตัด",
    "วงเวียน",
    "intersection",
    "junction",
    "roundabout"
]

STRAIGHT_KEYWORDS = [
    "ถนน",
    "สะพาน",
    "สะพานลอย",
    "ทางคู่ขนาน",
    "ทางขึ้น",
    "ทางลง",
    "แนวถนน",
    "ถนนคู่ขนาน",
    "ถนนเลียบ",
    "road",
    "street",
    "footbridge",
    "expressway",
    "highway"
]


def contains_keyword(text, keywords):

    if pd.isna(text):
        return False

    text = str(text).lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def classify_road_type(location, location_detail):

    location = (
        "" if pd.isna(location)
        else str(location).lower()
    )

    location_detail = (
        "" if pd.isna(location_detail)
        else str(location_detail).lower()
    )

    # ตรวจชื่อสถานที่ก่อน
    if contains_keyword(location, INTERSECTION_KEYWORDS):
        return "intersection"

    if contains_keyword(location, STRAIGHT_KEYWORDS):
        return "straight"

    # ถ้าชื่อสถานที่ไม่ชัดเจน ให้ตรวจรายละเอียด
    if contains_keyword(location_detail, INTERSECTION_KEYWORDS):
        return "intersection"

    if contains_keyword(location_detail, STRAIGHT_KEYWORDS):
        return "straight"

    return None


# ==========================================
# 3. Create study boundary
# ==========================================

def create_study_bbox(df):

    """
    สร้างกรอบพื้นที่ศึกษาอัตโนมัติจากพิกัดกล้องทั้งหมด
    และขยายกรอบออกตาม BBOX_PADDING
    """

    min_lon = df["longitude"].min() - BBOX_PADDING
    max_lon = df["longitude"].max() + BBOX_PADDING

    min_lat = df["latitude"].min() - BBOX_PADDING
    max_lat = df["latitude"].max() + BBOX_PADDING

    study_bbox = box(
        min_lon,
        min_lat,
        max_lon,
        max_lat
    )

    print("\nStudy boundary:")
    print(f"Longitude: {min_lon:.6f} to {max_lon:.6f}")
    print(f"Latitude : {min_lat:.6f} to {max_lat:.6f}")

    return study_bbox


# ==========================================
# 4. Download Chao Phraya river from OSM
# ==========================================

def download_river(study_bbox):

    min_lon, min_lat, max_lon, max_lat = study_bbox.bounds

    # Overpass API ใช้ลำดับ south, west, north, east
    query = f"""
    [out:json][timeout:180];

    way["waterway"="river"]
        ["name"~"Chao Phraya|เจ้าพระยา",i]
        ({min_lat},{min_lon},{max_lat},{max_lon});

    out geom;
    """

    print("\nDownloading Chao Phraya river from OpenStreetMap...")

    endpoints = [OVERPASS_URL] + [
        url for url in OVERPASS_FALLBACK_URLS
        if url != OVERPASS_URL
    ]

    data = None
    last_error = None

    for endpoint_index, endpoint in enumerate(endpoints):

        attempts = 2 if endpoint_index == 0 else 1

        for attempt in range(attempts):

            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=240,
                    headers={
                        "User-Agent": "CCTV-Vehicle-Analysis/1.0"
                    }
                )

                response.raise_for_status()
                data = response.json()
                print(f"Overpass response received from {endpoint}")
                break

            except requests.RequestException as error:
                last_error = error
                print(
                    f"Overpass request failed for {endpoint} "
                    f"(attempt {attempt + 1}/{attempts}): {error}"
                )

                if attempt + 1 < attempts:
                    time.sleep(2)

        if data is not None:
            break

    if data is None:
        raise RuntimeError(
            "ไม่สามารถดาวน์โหลดข้อมูลแม่น้ำเจ้าพระยาจาก Overpass API "
            "ได้ทั้ง endpoint หลักและ endpoint สำรอง กรุณาลองใหม่ภายหลัง"
        ) from last_error

    elements = data.get("elements", [])

    river_lines = []

    for element in elements:

        geometry = element.get("geometry")

        if not geometry or len(geometry) < 2:
            continue

        coordinates = [
            (point["lon"], point["lat"])
            for point in geometry
        ]

        river_lines.append(
            LineString(coordinates)
        )

    if not river_lines:
        raise ValueError(
            "ไม่พบเส้นแม่น้ำเจ้าพระยาในพื้นที่ศึกษา "
            "กรุณาตรวจสอบ Overpass API หรือข้อมูล OSM"
        )

    print(f"Downloaded {len(river_lines)} river ways")

    return river_lines


# ==========================================
# 5. Build Phra Nakhon polygon
# ==========================================

def build_phra_nakhon_polygon(
    study_bbox,
    river_lines
):

    """
    ใช้เส้นแม่น้ำแบ่งกรอบพื้นที่ศึกษาออกเป็นสองฝั่ง

    จากนั้นเลือก Polygon ที่ครอบคลุมจุดอ้างอิง
    ฝั่งพระนคร
    """

    anchor = Point(
        PHRA_NAKHON_LON,
        PHRA_NAKHON_LAT
    )

    if not study_bbox.covers(anchor):
        raise ValueError(
            "จุดอ้างอิงฝั่งพระนครอยู่นอกกรอบพื้นที่ศึกษา"
        )

    # รวมเส้นแม่น้ำที่ดาวน์โหลดมา
    merged_river = unary_union(river_lines)

    # ตัดเส้นแม่น้ำให้อยู่ภายในกรอบพื้นที่ศึกษา
    clipped_river = merged_river.intersection(study_bbox)

    if clipped_river.is_empty:
        raise ValueError(
            "ไม่พบเส้นแม่น้ำภายในกรอบพื้นที่ศึกษา"
        )

    # นำเส้นแม่น้ำมาต่อกับขอบกรอบพื้นที่
    # เพื่อแบ่งกรอบออกเป็น Polygon
    boundary_lines = unary_union([
        study_bbox.boundary,
        clipped_river
    ])

    polygons = list(
        polygonize(boundary_lines)
    )

    if not polygons:
        raise ValueError(
            "ไม่สามารถสร้าง Polygon จากเส้นแม่น้ำได้ "
            "เส้นแม่น้ำอาจไม่ต่อเนื่องหรือไม่ตัดผ่านกรอบพื้นที่"
        )

    # เลือก Polygon ที่ครอบคลุมจุดอ้างอิงฝั่งพระนคร
    phra_nakhon_polygons = [
        polygon
        for polygon in polygons
        if polygon.covers(anchor)
    ]

    if len(phra_nakhon_polygons) != 1:
        raise ValueError(
            "ไม่สามารถระบุ Polygon ฝั่งพระนครได้อย่างชัดเจน "
            "กรุณาตรวจสอบเส้นแม่น้ำและจุดอ้างอิง"
        )

    phra_nakhon_polygon = phra_nakhon_polygons[0]

    print("\nPolygon created successfully")
    print(f"Total polygons: {len(polygons)}")
    print("Phra Nakhon polygon identified")

    return phra_nakhon_polygon


# ==========================================
# 6. Classify camera side
# ==========================================

def classify_side(
    latitude,
    longitude,
    study_bbox,
    phra_nakhon_polygon
):

    point = Point(
        longitude,
        latitude
    )

    # ตรวจว่าพิกัดกล้องอยู่ในพื้นที่ศึกษาหรือไม่
    if not study_bbox.covers(point):
        return None

    # อยู่ใน Polygon ฝั่งพระนคร
    if phra_nakhon_polygon.covers(point):
        return "PhraNakhon"

    # อยู่ในอีกฝั่งของแม่น้ำภายในกรอบศึกษา
    return "Thonburi"


# ==========================================
# 7. Main
# ==========================================

def main():

    print("Reading cameras.csv...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "camera_id",
        "location",
        "location_detail",
        "latitude",
        "longitude"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df["camera_id"].duplicated().any():
        raise ValueError("พบ camera_id ซ้ำ")

    # ตรวจสอบพิกัด
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    invalid_coordinates = df[
        df["latitude"].isna()
        | df["longitude"].isna()
    ]

    if not invalid_coordinates.empty:
        raise ValueError(
            "พบกล้องที่ไม่มีพิกัดถูกต้อง:\n"
            + invalid_coordinates[
                ["camera_id", "latitude", "longitude"]
            ].to_string(index=False)
        )

    # สร้างกรอบพื้นที่ศึกษา
    study_bbox = create_study_bbox(df)

    # ดาวน์โหลดแนวแม่น้ำ
    river_lines = download_river(study_bbox)

    # สร้าง Polygon ฝั่งพระนคร
    phra_nakhon_polygon = build_phra_nakhon_polygon(
        study_bbox,
        river_lines
    )

    # จำแนกฝั่ง
    print("\nClassifying camera sides...")

    df["side"] = df.apply(
        lambda row: classify_side(
            row["latitude"],
            row["longitude"],
            study_bbox,
            phra_nakhon_polygon
        ),
        axis=1
    )

    if df["side"].isna().any():
        raise ValueError(
            "พบกล้องที่อยู่นอกกรอบพื้นที่ศึกษา"
        )

    # จำแนกประเภทถนน
    # Classify road types
    print("\nClassifying road types...")
    df["road_type"] = df.apply(
        lambda row: classify_road_type(
            row["location"],
            row["location_detail"]
        ),
        axis=1
    )

    # Check unclassified road types
    unclassified = df[df["road_type"].isna()].copy()

    if not unclassified.empty:
        print("\nSkipping cameras with unclassified road_type:")
        print(
            unclassified[
                ["camera_id", "location", "location_detail"]
            ].to_string(index=False)
        )

        print(f"\nSkipped cameras: {len(unclassified)}")

        # Remove cameras that cannot be classified
        df = df[df["road_type"].notna()].copy()

    else:
        print("All cameras have been classified successfully.")

    # บันทึกผลลัพธ์
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nClassification completed!")

    print(f"Output: {OUTPUT_FILE}")

    print("\nSide summary:")
    print(df["side"].value_counts())

    print("\nRoad type summary:")
    print(df["road_type"].value_counts())

    print("\nPreview:")
    print(
        df[
            ["camera_id", "location", "side", "road_type"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()