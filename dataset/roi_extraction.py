import numpy as np
import SimpleITK
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import warnings


from mindful_core.utils.parsing import parse_last_number


# region Load
def get_patient_folders(input_image_folder: Path) -> dict[str, Path]:
    patient_folders = {}
    for path in input_image_folder.iterdir():
        if not path.is_dir():
            continue

        patient_id = parse_last_number(path.stem)
        if patient_id is not None:
            patient_folders[str(patient_id)] = path
    return patient_folders


def get_points_filepaths(points_folder: Path) -> dict[str, dict[str, Path]]:
    points_filepaths = {}
    for path in points_folder.glob("*.txt"):
        patient_id, _, lesion_index, _ = path.stem.split("_")
        if patient_id not in points_filepaths:
            points_filepaths[patient_id] = {}
        points_filepaths[patient_id][lesion_index] = path
    return points_filepaths


def load_patient_phases(patient_folder: Path) -> dict[int, SimpleITK.Image]:
    patient_images = {}
    for path in patient_folder.glob("*.mha"):
        expected_base_name = "{}_t".format(patient_folder.stem)
        file_id = path.stem[len(expected_base_name):]
        if not file_id.isdigit():
            continue

        patient_images[int(file_id)] = SimpleITK.ReadImage(path.as_posix())
    return patient_images


def load_all_patients_phases(patient_folders: dict[str, Path]) -> dict[str, dict[int, SimpleITK.Image]]:
    progress_bar = tqdm(patient_folders, desc="Loading patient images (4D)...")

    patients_images: dict[str, dict[int, SimpleITK.Image]] = {}
    for patient_id in progress_bar:
        patient_folder = patient_folders[patient_id]
        progress_bar.set_description("Loading patient images... ({})".format(patient_folder))

        try:
            patients_images[patient_id] = load_patient_phases(patient_folder)
        except KeyboardInterrupt:
            break

    return patients_images

def is_phases_folder(path: Path) -> bool:
    if not path.is_dir():
        return False
    
    for filepath in path.iterdir():
        if filepath.match("*_t.mha"):
            return True
        
    return False

def find_vista_images(folder: Path) -> dict[str, Path] | None:
    vista_filepaths = list(folder.glob("p*_VISTA.mha"))
    if len(vista_filepaths) == 0:
        return None

    # Filepath form: pXXX_VISTA.mha (where XXX is the patient's ID)    
    result = {filepath.stem.split("_")[0][1:]: filepath for filepath in vista_filepaths}
    return result

def load_patients_vista(folder: Path) -> dict[str, SimpleITK.Image] | None:
    vista_filepaths = find_vista_images(folder)
    if len(vista_filepaths) == 0:
        return None
    
    progress_bar = tqdm(vista_filepaths.items(), desc="Loading patient images (VISTA)...")

    images = {}
    for patient_id, filepath in progress_bar:
        progress_bar.set_description("Loading patient images... ({})".format(patient_id))
        images[patient_id] = SimpleITK.ReadImage(filepath.as_posix())

    return images

def find_patient_images(input_image_folder: Path) -> dict[str, Path]:
    folder_content = list(input_image_folder.iterdir())

    by_phase_arborescence = any([is_phases_folder(filepath) for filepath in folder_content])
    if by_phase_arborescence:
        patient_paths = get_patient_folders(input_image_folder)
        return patient_paths
    
    vista_paths = find_vista_images(input_image_folder)
    if vista_paths is not None:
        return vista_paths
    
    raise RuntimeError("Could not determine how to parse patient IDs for loading images.")


def find_and_load_patient_images(input_image_folder: Path) -> dict[str, dict[int, SimpleITK.Image]]:
    folder_content = list(input_image_folder.iterdir())

    by_phase_arborescence = any([is_phases_folder(filepath) for filepath in folder_content])
    if by_phase_arborescence:
        patient_folders = get_patient_folders(input_image_folder)
        patients_images = load_all_patients_phases(patient_folders)
        return patients_images
    
    vista_images = load_patients_vista(input_image_folder)
    if vista_images is not None:
        patients_images = {patient_id: {0: image} for patient_id, image in vista_images.items()}
        return patients_images
    
    raise RuntimeError("Could not determine how to parse patient IDs for loading images.")

def load_patient_points(patient_points_filepaths: dict[str, Path]) -> dict[str, np.ndarray]:
    patient_points = {}
    for lesion_id, points_filepath in patient_points_filepaths.items():
        with open(points_filepath, "r") as points_file:
            lines = [line.rstrip() for line in points_file]
            if len(lines) < 2:
                raise RuntimeError("Points file is empty ({})".format(points_filepath.as_posix()))

            points_count = int(lines.pop(0))
            points = [np.asarray(line.split(" "), dtype=np.float32) for _, line in zip(range(points_count), lines)]
            patient_points[lesion_id] = np.stack(points, axis=0)
    return patient_points


def load_all_patients_points(patients_points_filepaths: dict[str, dict[str, Path]]) -> dict[str, dict[str, np.ndarray]]:
    patients_points = {patient_id: load_patient_points(patients_points_filepaths[patient_id])
                       for patient_id in patients_points_filepaths}
    return patients_points


def points_filepaths_to_short_labels(patients_points_filepaths: dict[str, dict[str, Path]]
                                     ) -> dict[str, str]:
    return {"{}_{}".format(patient_id, lesion_id): points_filepath.stem.split("_")[-1]
            for patient_id, lesions_points_filepaths in patients_points_filepaths.items()
            for lesion_id, points_filepath in lesions_points_filepaths.items()}


# endregion

# region Save
def get_full_label(short_label: str):
    label_map = {
        "M": "positive",
        "B": "negative",
        "G": "lymphnode",
        "IMLN": "lymphnode",
        "ALN": "lymphnode",
    }
    
    if short_label not in label_map:
        known_labels = ", ".join(label_map)
        raise ValueError("Unknown short label `{}`. Known labels are: {}".format(short_label, known_labels))

    return label_map[short_label]


def save_roi(images: dict[int, SimpleITK.Image] | list[SimpleITK.Image] | SimpleITK.Image,
             root_output_folder: Path,
             sample_name: str,
             version_id: str = None,
             image_names: list[str] | str = None,
             ) -> list[Path] | Path:
    # path : root/label/patient-id_lesion_lesion-id_point-id_short-label/phase_image-id.mha

    if (version_id is not None) and (len(version_id) > 0):
        sample_name = "{}_version_{}".format(sample_name, version_id)

    if isinstance(images, SimpleITK.Image):
        if not root_output_folder.exists():
            root_output_folder.mkdir(parents=True)

        output_filepath = root_output_folder / (sample_name + ".mha")
        SimpleITK.WriteImage(images, output_filepath.as_posix())
        return output_filepath

    else:
        outputs: list[Path] = []

        output_folder = root_output_folder / sample_name
        if not output_folder.exists():
            output_folder.mkdir(parents=True)

        if isinstance(images, dict):
            image_ids = sorted(list(images.keys()))
        else:
            image_ids = range(len(images))

        if image_names is None:
            image_names = image_ids

        if isinstance(image_names, str):
            if len(image_names) == 0:
                image_names = image_ids
            else:
                image_names = ["{}_{}".format(image_names, image_id) for image_id in image_ids]

        for image_id, image_name in zip(image_ids, image_names):
            output_filepath = output_folder / "{}.mha".format(image_name)
            outputs.append(output_filepath)
            SimpleITK.WriteImage(images[image_id], output_filepath.as_posix())

        return outputs


def save_rois(output_images_folder: Path,
              rois: dict[str, list[SimpleITK.Image]] | dict[str, SimpleITK.Image],
              short_labels: dict[str, str] | pd.Series | None = None,
              use_progress_bar: bool = True,
              image_names: list[str] | str | dict[str, list[str]] | dict[str, str] | None = None,
              ) -> dict[str, list[Path]] | dict[str, Path]:
    extracted_roi_paths: dict[str, list[Path]] | dict[str, Path] = {}
    progress_bar = tqdm(rois, desc="Saving ROIs to disk...", disable=not use_progress_bar)

    for roi_id in progress_bar:
        progress_bar.set_description("Saving ROIs to disk... ({})".format(roi_id))

        try:
            sub_ids = roi_id.split("_")
            if (len(sub_ids) >= 3) and (short_labels is not None):
                patient_id, lesion_index, point_index, *version_ids = roi_id.split("_")
                lesion_id = "{}_{}".format(patient_id, lesion_index)
                short_label = short_labels[lesion_id]
                sample_name = "{}_lesion_{}_{}_{}".format(patient_id, lesion_index, point_index, short_label)
                version_id = "_".join(version_ids)
                full_label = get_full_label(short_label)
                sample_class_folder = output_images_folder / full_label
            else:
                sample_name = roi_id
                version_id = None
                sample_class_folder = output_images_folder

            output_paths = save_roi(images=rois[roi_id],
                                    root_output_folder=sample_class_folder,
                                    sample_name=sample_name,
                                    version_id=version_id,
                                    image_names=image_names)

            extracted_roi_paths[roi_id] = output_paths

        except KeyboardInterrupt:
            # Catching KeyboardInterrupt to stop the process earlier (mostly for debug)
            break

    return extracted_roi_paths


# endregion

# region Process
def to_images_sub(images: dict[int, SimpleITK.Image]) -> dict[int, SimpleITK.Image]:
    if 0 not in images:
        raise ValueError("Missing image 0 needed for image subtraction.")
    base_image = images[0]

    sub_images = {}

    for image_id, image in images.items():
        if image_id == 0:
            continue

        sub_images[image_id - 1] = image - base_image

    return sub_images


def extract_roi(images: dict[int, SimpleITK.Image], point: np.ndarray, roi_size: float) -> dict[int, SimpleITK.Image]:
    if point.shape != (3,):
        raise ValueError("Expected point to be a vector with length 3 (x, y, z), got an array with shape {}."
                         .format(point.shape))

    identity = SimpleITK.Transform(3, SimpleITK.sitkIdentity)
    output_direction = np.reshape(np.eye(3, 3), newshape=[9])

    output_images = {}
    for image_id in images:
        image = images[image_id]
        spacing = np.asarray(image.GetSpacing(), dtype=point.dtype)
        output_size = roi_size / spacing + 0.5
        output_origin = point - roi_size * 0.5

        # region Convert np array to built-in types for SimpleITK
        output_size = [int(_x) for _x in output_size]
        spacing = [float(_x) for _x in spacing]
        output_origin = [float(_x) for _x in output_origin]
        # endregion
        output = SimpleITK.Resample(image, output_size, identity, SimpleITK.sitkLinear, output_origin, spacing,
                                    output_direction)
        output_images[image_id] = output
    return output_images


def extract_all_rois(patients_images: dict[str, dict[int, SimpleITK.Image]],
                     patients_points: dict[str, dict[str, np.ndarray]],
                     use_sub: bool,
                     roi_size: float,
                     verbose=True,
                     out_patients_missing_points: list = None
                     ) -> dict[str, list[SimpleITK.Image]]:
    progress_bar = tqdm(patients_images, desc="Extracting ROIs...")

    extracted_rois: dict[str, list[SimpleITK.Image]] = {}
    for patient_id in progress_bar:
        try:
            if patient_id not in patients_points:
                if verbose:
                    warnings.warn("Patient `{}` has no matching points file.".format(patient_id))
                if out_patients_missing_points is not None:
                    out_patients_missing_points.append(patient_id)
                continue

            progress_bar.set_description("Extracting ROIs... ({})".format(patient_id))
            patient_images = patients_images[patient_id]
            if use_sub:
                patient_images = to_images_sub(patient_images)

            for lesion_index, lesion_points in patients_points[patient_id].items():
                lesion_id = "{}_{}".format(patient_id, lesion_index)
                for point_index, point in enumerate(lesion_points):
                    point: np.ndarray
                    roi = extract_roi(patient_images, point, roi_size)
                    roi_id = "{}_{}".format(lesion_id, point_index)
                    extracted_rois[roi_id] = list(roi.values())
        except KeyboardInterrupt:
            # Catching KeyboardInterrupt to stop the process earlier (mostly for debug)
            break

    return extracted_rois


# endregion


