import json
import argparse
import pandas as pd
from pathlib import Path


def extract_name_from_mask(mask_path: str) -> str:
    """
    Extract lesion name from mask path.

    Format:
        {ID}_{lesion_index}_N.ext

    ID may contain underscores.
    """
    stem = Path(mask_path).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected mask filename format: {mask_path}")

    lesion_index = parts[-2]
    lesion_id = "_".join(parts[:-2])

    return f"{lesion_id}_{lesion_index}_0"


def csv_to_jsons(
    csv_path: str,
    output_char_json: str,
    output_pos_json: str,
    left_nipple=None,
    right_nipple=None
):
    csv_path = Path(csv_path)
    output_char_json = Path(output_char_json)
    output_pos_json = Path(output_pos_json)

    df = pd.read_csv(csv_path)

    lesions = []
    points = []

    # Optional nipple points (added first like your example)
    if left_nipple is not None:
        points.append({
            "name": "nipple_left",
            "point": left_nipple,
            "probability": 1.0
        })

    if right_nipple is not None:
        points.append({
            "name": "nipple_right",
            "point": right_nipple,
            "probability": 1.0
        })

    for idx, row in df.iterrows():
        try:
            name = extract_name_from_mask(row["mask_filename"])
        except Exception as e:
            print(f"[WARNING] Skipping row {idx}: {e}")
            continue

        try:
            geometry = {
                "volume": float(row["volume"]),
                "flatness": float(row["flatness"]),
                "elongation": float(row["elongation"]),
                "bounding_box_size_x": float(row["bounding_box_size_x"]),
                "bounding_box_size_y": float(row["bounding_box_size_y"]),
                "bounding_box_size_z": float(row["bounding_box_size_z"]),
                "bounding_box_center_x": float(row["bounding_box_center_x"]),
                "bounding_box_center_y": float(row["bounding_box_center_y"]),
                "bounding_box_center_z": float(row["bounding_box_center_z"]),
                "equivalent_ellipsoid_diameter_1": float(row["equivalent_ellipsoid_diameter_1"]),
                "equivalent_ellipsoid_diameter_2": float(row["equivalent_ellipsoid_diameter_2"]),
                "equivalent_ellipsoid_diameter_3": float(row["equivalent_ellipsoid_diameter_3"]),
            }
        except Exception as e:
            print(f"[WARNING] Invalid geometry at row {idx}: {e}")
            continue

        # --- Characteristics JSON ---
        lesions.append({
            "name": name,
            "geometry": geometry
        })

        # --- Positions JSON (gravity center) ---
        point = [
            float(row["center_of_gravity_x"]),
            float(row["center_of_gravity_y"]),
            float(row["center_of_gravity_z"])
        ]

        points.append({
            "name": name,
            "point": point,
            "probability": 1.0
        })

    # --- Write characteristics JSON ---
    char_result = {
        "lesions": lesions,
        "compute_lesion_geometry": False
    }

    output_char_json.parent.mkdir(parents=True, exist_ok=True)
    with output_char_json.open("w", encoding="utf-8") as f:
        json.dump(char_result, f, indent=4)

    # --- Write positions JSON ---
    pos_result = {
        "name": "Points of interest",
        "type": "Multiple points",
        "points": points,
        "version": {
            "major": 1,
            "minor": 0
        }
    }

    output_pos_json.parent.mkdir(parents=True, exist_ok=True)
    with output_pos_json.open("w", encoding="utf-8") as f:
        json.dump(pos_result, f, indent=4)

    print(f"Characteristics JSON written to: {output_char_json}")
    print(f"Positions JSON written to: {output_pos_json}")
    print(f"Total lesions exported: {len(lesions)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert lesion CSV into characteristics and positions JSON files."
    )

    parser.add_argument("--csv_path", required=True, help="Input CSV file")

    parser.add_argument(
        "--output_char_json",
        required=True,
        help="Output JSON for lesion characteristics"
    )

    parser.add_argument(
        "--output_pos_json",
        required=True,
        help="Output JSON for lesion positions"
    )

    parser.add_argument(
        "--left_nipple",
        nargs=3,
        type=float,
        help="Left nipple coordinates: x y z"
    )

    parser.add_argument(
        "--right_nipple",
        nargs=3,
        type=float,
        help="Right nipple coordinates: x y z"
    )

    args = parser.parse_args()

    csv_to_jsons(
        csv_path=args.csv_path,
        output_char_json=args.output_char_json,
        output_pos_json=args.output_pos_json,
        left_nipple=args.left_nipple,
        right_nipple=args.right_nipple
    )


if __name__ == "__main__":
    main()