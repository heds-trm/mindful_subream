from pathlib import Path
import argparse

from mindful_core.utils.misc import load_table

TYPE_TO_LABEL = {
    "B": "B",
    "M": "M",
    "MLN": "M",
    "G": "G",
    "IMLN": "G",
    "ALN": "G",
}


def get_expectations(data_breast_path: Path):
    data_breast = load_table(data_breast_path)
    if data_breast is None:
        raise FileNotFoundError("Could not load the `data breast` table from {}".format(data_breast_path))

    if "Code" not in data_breast.columns:
        raise KeyError("The `Code` column is required in the `data breast` table.")

    n_lesion_aliases = ["n° lésion", "N° lésion"]
    n_lesion_column = None
    for alias in n_lesion_aliases:
        if alias in data_breast.columns:
            n_lesion_column = alias
            break

    if n_lesion_column is None:
        raise KeyError("No valid lesion index column found in `data breast` table. Got {} but expected any of {}".
                       format(data_breast.columns, n_lesion_aliases))

    expected_lesions: dict[int, list[int]] = {}
    expected_lesions_labels: dict[str, str] = {}

    for _, row in data_breast.iterrows():
        lesion_type = row["type"]
        if not isinstance(lesion_type, str):
            continue

        patient_code = int(row["Code"])
        lesion_index = int(row[n_lesion_column])
        lesion_id = "{}_{}".format(patient_code, lesion_index)

        if patient_code not in expected_lesions:
            expected_lesions[patient_code] = []
        expected_lesions[patient_code].append(lesion_index)

        lesion_label = TYPE_TO_LABEL[lesion_type]

        expected_lesions_labels[lesion_id] = lesion_label

    expected_lesions_count = {patient_code: len(lesion_indices) for patient_code, lesion_indices in
                              expected_lesions.items()}

    return expected_lesions, expected_lesions_count, expected_lesions_labels


def list_ultrafast_images(ultrafast_images_root: Path, code_prefix: str) -> dict[int, Path]:
    if len(code_prefix) == 0:
        code_prefix = "_"

    ultrafast_images: dict[int, Path] = {}

    folder_prefix = "Patient{}".format(code_prefix)
    pattern = folder_prefix + "*"
    prefix_length = len(folder_prefix)

    for path in ultrafast_images_root.glob(pattern):
        if not path.is_dir():
            continue

        patient_code = int(path.name[prefix_length:])
        ultrafast_images[patient_code] = path

    return ultrafast_images


def list_t2_images(t2_images_root: Path, code_prefix: str) -> dict[int, Path]:
    if len(code_prefix) == 0:
        code_prefix = "p"

    t2_images: dict[int, Path] = {}

    pattern = "{}*.mha".format(code_prefix)
    prefix_length = len(code_prefix)

    for path in t2_images_root.glob(pattern):
        patient_id = path.stem.split("_")[0]
        patient_code = int(patient_id[prefix_length:])

        t2_images[patient_code] = path

    return t2_images


def list_roi_masks(roi_masks_root: Path, code_prefix: str) -> tuple[dict[str, Path], dict[str, str]]:
    roi_masks: dict[str, Path] = {}
    roi_labels: dict[str, str] = {}

    pattern = "{}*_lesion_*_*.mha".format(code_prefix)
    prefix_length = len(code_prefix)

    for path in roi_masks_root.glob(pattern):
        patient_id, _, _, lesion_index, lesion_label = path.stem.split("_")
        patient_code = int(patient_id[prefix_length:])

        lesion_id = "{}_{}".format(patient_code, lesion_index)
        roi_masks[lesion_id] = path
        roi_labels[lesion_id] = lesion_label

    return roi_masks, roi_labels


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_breast", required=True, type=str)
    # arg_parser.add_argument("--ultrafast_images", required=True, type=str)
    # arg_parser.add_argument("--t2_images", required=True, type=str)
    arg_parser.add_argument("--roi_masks", required=True, type=str)
    arg_parser.add_argument("--code_prefix", type=str, default="")

    args = arg_parser.parse_args()
    data_breast_path = Path(args.data_breast)
    # ultrafast_images_root = Path(args.ultrafast_images)
    # t2_images_root = Path(args.t2_images)
    roi_masks_root = Path(args.roi_masks)
    code_prefix: str = args.code_prefix

    expected_lesions, expected_lesions_count, expected_lesions_labels = get_expectations(data_breast_path)

    # ultrafast_images_paths = list_ultrafast_images(ultrafast_images_root, code_prefix)
    # t2_images_paths = list_t2_images(t2_images_root, code_prefix)
    roi_masks_paths, roi_labels = list_roi_masks(roi_masks_root, code_prefix)

    missing_expected_lesion_ids = []
    wrong_labels = {}
    for lesion_id, expected_lesion_label in expected_lesions_labels.items():
        if lesion_id not in roi_labels:
            missing_expected_lesion_ids.append(lesion_id)
            continue

        if expected_lesion_label != roi_labels[lesion_id]:
            wrong_labels[lesion_id] = (expected_lesion_label, roi_labels[lesion_id])

    if len(missing_expected_lesion_ids) > 0:
        print("You are missing the following lesion masks:")
        print(missing_expected_lesion_ids)
    else:
        print("No missing lesion masks were found!")

    if len(wrong_labels) > 0:
        print("The following lesions have conflicting labels:")
        print(wrong_labels)
    else:
        print("No conflict between filenames and labels have been found!")


if __name__ == "__main__":
    main()
