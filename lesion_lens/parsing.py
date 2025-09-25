import numpy as np
from pathlib import Path

from mindful_core.utils.misc import load_json


def parse_suspected_lesions_positions(suspected_lesions_positions_path: str | Path
                                      ) -> dict[str, np.ndarray]:
    suspected_lesions_data = load_json(suspected_lesions_positions_path)
    points: list[dict[str, str | list[float]]] = suspected_lesions_data["points"]

    suspected_lesions_positions: dict[str, np.ndarray] = {}
    for point_data in points:
        point_name = point_data["name"]
        point_array = np.asarray(point_data["point"], np.float32)
        suspected_lesions_positions[point_name] = point_array

    return suspected_lesions_positions


def parse_scalar_data(patient_information_path: str | Path,
                      lesion_information_path: str | Path,
                      suspected_lesions_positions: dict[str, np.ndarray]
                      ) -> dict[str, np.ndarray] | None:
    patient_information_path = Path(patient_information_path)
    lesion_information_path = Path(lesion_information_path)
    if (not patient_information_path.exists()) or (not lesion_information_path.exists()):
        return None

    patient_information = load_json(patient_information_path)
    age = patient_information["age"]

    lesion_information = load_json(lesion_information_path)
    compute_lesion_geometry = lesion_information["compute_lesion_geometry"]
    if compute_lesion_geometry:
        raise NotImplementedError("`compute_lesion_geometry=True` is not supported yet.")

    lesions_data: list[dict[str, str | dict[str, float]]] = lesion_information["lesions"]
    lesions_scalar_data: dict[str, np.ndarray] = {}
    for lesion_data in lesions_data:
        name: str = lesion_data["name"]
        geometry: dict[str, float] = lesion_data["geometry"]
        position: np.ndarray = suspected_lesions_positions[name]
        lesion_scalars = [
            age,

            geometry["bounding_box_center_x"],
            geometry["bounding_box_center_y"],
            geometry["bounding_box_center_z"],

            geometry["bounding_box_size_x"],
            geometry["bounding_box_size_y"],
            geometry["bounding_box_size_z"],

            geometry["volume"],

            position[0],
            position[1],
            position[2],

            geometry["elongation"],

            geometry["equivalent_ellipsoid_diameter_1"],
            geometry["equivalent_ellipsoid_diameter_2"],
            geometry["equivalent_ellipsoid_diameter_3"],

            geometry["flatness"],
        ]

        lesions_scalar_data[name] = np.asarray(lesion_scalars, np.float32)

    return lesions_scalar_data


def parse_categorical_data(patient_information_path: str | Path) -> dict[str, str] | None:
    patient_information_path = Path(patient_information_path)
    if not patient_information_path.exists():
        return None

    patient_information = load_json(patient_information_path)
    if "age" in patient_information:
        patient_information.pop("age")
    return patient_information
