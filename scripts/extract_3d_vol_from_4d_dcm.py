import argparse
import os
import SimpleITK as sitk
from pathlib import Path
from collections import defaultdict


def sort_dicom_files_by_position(file_list):
    """
    Sort DICOM slices using ImagePositionPatient (Z coordinate).
    This is essential to avoid head-feet flipping.
    """
    reader = sitk.ImageFileReader()

    def get_z(f):
        reader.SetFileName(f)
        reader.ReadImageInformation()

        if not reader.HasMetaDataKey("0020|0032"):
            raise ValueError(f"Missing ImagePositionPatient in {f}")

        pos = reader.GetMetaData("0020|0032")
        z = float(pos.split("\\")[-1])
        return z

    return sorted(file_list, key=get_z)


def discover_dicom_series(dicom_folder):
    """Recursively discover DICOM series in a folder tree."""
    reader = sitk.ImageSeriesReader()
    series_entries = []

    for dirpath, _, filenames in os.walk(dicom_folder):
        if not filenames:
            continue

        ids = reader.GetGDCMSeriesIDs(dirpath) or []
        for series_id in ids:
            series_entries.append((dirpath, series_id))

    return series_entries


def load_4d_dicom(dicom_folder):
    """
    Load DICOM data supporting:
    - multiple 3D series (one per phase)
    - single series with temporal tags (true 4D)
    """
    reader = sitk.ImageSeriesReader()
    series_entries = discover_dicom_series(dicom_folder)

    if not series_entries:
        raise ValueError("No DICOM series found.")

    # -------------------------
    # CASE 1: multiple series
    # -------------------------
    if len(series_entries) > 1:
        print(f"Detected {len(series_entries)} series → assuming one phase per series")

        volumes = []
        for series_dir, series_id in sorted(series_entries):
            files = reader.GetGDCMSeriesFileNames(series_dir, series_id)
            files = sort_dicom_files_by_position(files)

            reader.SetFileNames(files)
            volumes.append(reader.Execute())

        return sitk.JoinSeries(volumes)

    # -------------------------
    # CASE 2: single series
    # -------------------------
    series_dir, series_id = series_entries[0]
    print(f"Single series detected in {series_dir} → grouping by temporal tags")

    dicom_files = reader.GetGDCMSeriesFileNames(series_dir, series_id)
    file_reader = sitk.ImageFileReader()

    phase_dict = defaultdict(list)

    for f in dicom_files:
        file_reader.SetFileName(f)
        file_reader.ReadImageInformation()

        # TemporalPositionIdentifier (most common)
        if file_reader.HasMetaDataKey("0020|0100"):
            phase = file_reader.GetMetaData("0020|0100")
        else:
            phase = "0"

        phase_dict[phase].append(f)

    # If no temporal dimension → fallback to 3D
    if len(phase_dict) == 1:
        print("No temporal dimension detected → returning single 3D as 4D")

        files = sort_dicom_files_by_position(dicom_files)
        reader.SetFileNames(files)
        img3d = reader.Execute()

        return sitk.JoinSeries([img3d])

    print(f"Detected {len(phase_dict)} phases")

    volumes = []
    for phase in sorted(phase_dict.keys()):
        files = sort_dicom_files_by_position(phase_dict[phase])

        reader.SetFileNames(files)
        volumes.append(reader.Execute())

    return sitk.JoinSeries(volumes)


def extract_phase(image_4d, phase_index):
    size = list(image_4d.GetSize())

    if len(size) != 4:
        raise ValueError("Image is not 4D.")

    n_phases = size[3]

    if phase_index < 0 or phase_index >= n_phases:
        raise ValueError(
            f"Phase index {phase_index} out of bounds (0 <= index < {n_phases})"
        )

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:3] + [0])
    extractor.SetIndex([0, 0, 0, phase_index])

    return extractor.Execute(image_4d)


def save_mha(image, output_path):
    output_path = Path(output_path)

    # Create output directory
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fix orientation (labels only)
    image = sitk.DICOMOrient(image, "LPS")

    # Save compressed
    sitk.WriteImage(image, str(output_path), useCompression=True)

    print(f"Saved (LPS, compressed): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a 3D phase from DICOM."
    )

    parser.add_argument("--dicom_folder", required=True)

    parser.add_argument("--phase_index", type=int, default=None)

    parser.add_argument("--last_phase", action="store_true")

    parser.add_argument("--output_mha", required=True)

    args = parser.parse_args()

    # --- Load 4D ---
    image_4d = load_4d_dicom(args.dicom_folder)

    size = list(image_4d.GetSize())
    if len(size) != 4:
        raise ValueError("Loaded image is not 4D.")

    n_phases = size[3]

    # --- Resolve phase ---
    if args.last_phase and args.phase_index is not None:
        raise ValueError("Use either --phase_index or --last_phase")

    if args.last_phase:
        phase_index = n_phases - 1
    elif args.phase_index is not None:
        phase_index = args.phase_index
    else:
        raise ValueError("Provide --phase_index or --last_phase")

    if phase_index < 0 or phase_index >= n_phases:
        raise ValueError(
            f"Phase index {phase_index} out of bounds (0 <= index < {n_phases})"
        )

    print(f"Using phase {phase_index} / {n_phases-1}")

    # --- Extract ---
    image_3d = extract_phase(image_4d, phase_index)

    # --- Save ---
    save_mha(image_3d, args.output_mha)


if __name__ == "__main__":
    main()