import torch
from monai.transforms import LoadImage, SaveImage, Resize
from monai.data import MetaTensor
import SimpleITK
import numpy as np
import cv2
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
import dataclasses
import copy
import datetime
import warnings
import shutil
from typing import Literal, Optional

from mindful_core.utils.tensor_utils import normalize
from mindful_core.utils.visualization import overlay_image, format_image
from mindful_core.utils.misc import find_first_path, write_json

from mindful_subream.lesion_lens.lesion_crop import LesionCrop

# region Classified Lesions Breast (JSON)
POINT_PREDICTIONS_SKELETON = {
    # "name": "Classified lesions in Breast",
    "type": "Multiple points",
    "points": [

    ],
    "version": {
        "major": 1,
        "minor": 0
    }
}


@dataclasses.dataclass
class NipplesPosition:
    left: np.ndarray
    right: np.ndarray

    @staticmethod
    def from_lesion_list(suspected_lesions_positions: dict[str, np.ndarray]
                         ) -> tuple[dict[str, np.ndarray], Optional["NipplesPosition"]]:
        suspected_lesions_positions = copy.copy(suspected_lesions_positions)

        nipples = {}
        for lesion_id in list(suspected_lesions_positions):
            if lesion_id.lower().startswith(("nipple", "mamelon")):
                nipples[lesion_id.lower()] = suspected_lesions_positions.pop(lesion_id)

        if len(nipples) == 0:
            return suspected_lesions_positions, None

        if len(nipples) != 2:
            raise RuntimeError("Unsupported number of nipples found: {}".format(len(nipples)))

        nipple_ids = tuple(nipples)
        nipple_a_id, nipple_b_id = nipple_ids[0].lower(), nipple_ids[1].lower()

        if nipple_a_id.endswith(("left", "gauche")) or nipple_b_id.endswith(("right", "droit", "droite")):
            left_nipple_position, right_nipple_position = nipples[nipple_a_id], nipples[nipple_b_id]
        elif nipple_b_id.endswith(("left", "gauche")) or nipple_a_id.endswith(("right", "droit", "droite")):
            left_nipple_position, right_nipple_position = nipples[nipple_b_id], nipples[nipple_a_id]
        else:
            nipple_a_position, nipple_b_position = nipples[nipple_a_id], nipples[nipple_b_id]
            if nipple_a_position[0] < nipple_b_position[0]:
                left_nipple_position, right_nipple_position = nipple_a_position, nipple_b_position
            else:
                left_nipple_position, right_nipple_position = nipple_b_position, nipple_a_position

        if left_nipple_position[0] == right_nipple_position[0]:
            raise ValueError("Left and right nipples have the same X position (left/right axis): {}".
                             format(left_nipple_position[0]))

        if not ((2 <= len(left_nipple_position) <= 3) and (2 <= len(right_nipple_position) <= 3)):
            raise ValueError("Nipple positions must contain 2 or 3 coordinates, got {} (left) and {} (right)".
                             format(len(left_nipple_position), len(right_nipple_position)))

        nipples_position = NipplesPosition(left_nipple_position, right_nipple_position)
        return suspected_lesions_positions, nipples_position

    def get_distance_to_closest_nipple(self,
                                       lesion_position: list[float] | np.ndarray
                                       ) -> tuple[float, Literal["left", "right"]]:
        lesion_position = np.asarray(lesion_position, dtype=np.float32)
        distance_to_left = np.sqrt(np.sum(np.square(lesion_position - self.left)))
        distance_to_right = np.sqrt(np.sum(np.square(lesion_position - self.right)))

        if distance_to_left < distance_to_right:
            return float(distance_to_left), "left"
        else:
            return float(distance_to_right), "right"


@dataclasses.dataclass
class LesionPointPrediction:
    name: str
    point: list[float]
    probability: float


def export_point_predictions(path: str | Path, lesions_point_predictions: list[LesionPointPrediction]) -> None:
    # noinspection PyTypeChecker
    lesions_point_predictions = [dataclasses.asdict(point_prediction) for point_prediction in lesions_point_predictions]
    output = copy.copy(POINT_PREDICTIONS_SKELETON)
    output["points"] = lesions_point_predictions
    write_json(path, output)


# endregion


# region Occlusion maps
def aggregate_occlusion_maps(occlusion_maps: dict[str, torch.Tensor],
                             lesion_crops: dict[str, LesionCrop],
                             ) -> torch.Tensor:
    output, output_size = None, None
    for lesion_id in lesion_crops:
        occlusion_map = occlusion_maps[lesion_id]
        lesion_crop = lesion_crops[lesion_id]

        if output is None:
            output_size = lesion_crop.original_image_size
            output = torch.zeros(size=output_size, dtype=torch.float32, device="cpu")

        resizer = Resize(lesion_crop.crop_spatial_shape)
        occlusion_map = resizer(occlusion_map).squeeze(0)

        if lesion_crop.pad_lesion:
            occlusion_map = lesion_crop.remove_padding(occlusion_map)

        if occlusion_map.numel() == 0:
            warnings.warn("Could not produce occlusion map for lesion `{}`: the occlusion map is empty. "
                          "Is your image correctly formatted ?".format(lesion_id))
        else:
            occlusion_map = occlusion_map / occlusion_map.max()
            output[lesion_crop.crop_slices] = occlusion_map

    return output


def export_occlusion_map(occlusion_maps: dict[str, torch.Tensor],
                         lesion_crops: dict[str, LesionCrop],
                         original_image_path: Path,
                         occlusion_map_filename: Path,
                         ) -> None:
    """
    Aggregates the different occlusion maps (one per lesion crop). Lesion crops are used for localizing where each
        occlusion map is located in the output. The original image is used for its metadata (notably its affine
        matrix for the position, spacing and orientation).
    The aggregated occlusion map is then saved to the given filepath.

    :param occlusion_maps:
    :param lesion_crops:
    :param original_image_path:
    :param occlusion_map_filename:
    :return:
    """
    occlusion_map = aggregate_occlusion_maps(occlusion_maps, lesion_crops)
    occlusion_map_filename.parent.mkdir(parents=True, exist_ok=True)

    loader = LoadImage()
    image = loader(original_image_path)
    occlusion_map = MetaTensor(occlusion_map, image.affine, image.meta)

    saver = SaveImage(output_dir=occlusion_map_filename.parent, output_postfix="", output_ext=".mha",
                      separate_folder=False, print_log=False)
    saver(occlusion_map, filename=occlusion_map_filename)


# endregion


# region PDF
UNKNOWN_VALUE = Literal["N/A"]


@dataclasses.dataclass
class LesionReportGeometry:
    volume: float | UNKNOWN_VALUE
    flatness: float | UNKNOWN_VALUE
    elongation: float | UNKNOWN_VALUE
    bounding_box_size_x: float | UNKNOWN_VALUE
    bounding_box_size_y: float | UNKNOWN_VALUE
    bounding_box_size_z: float | UNKNOWN_VALUE
    bounding_box_center_x: float | UNKNOWN_VALUE
    bounding_box_center_y: float | UNKNOWN_VALUE
    bounding_box_center_z: float | UNKNOWN_VALUE
    equivalent_ellipsoid_diameter_1: float | UNKNOWN_VALUE
    equivalent_ellipsoid_diameter_2: float | UNKNOWN_VALUE
    equivalent_ellipsoid_diameter_3: float | UNKNOWN_VALUE

    @classmethod
    def from_dict(cls, d: dict | None):
        if d is None:
            d = {}
        else:
            d = {key.lower(): value for key, value in d.items()}

        output = {}
        for field in cls.__annotations__.keys():
            if (field in d) and isinstance(d[field], (float, int)):
                output[field] = round(d[field], 1)
            else:
                output[field] = "N/A"

        return LesionReportGeometry(**output)


@dataclasses.dataclass
class LesionReportData:
    point: list[float]
    geometry: LesionReportGeometry
    closest_nipple: Literal["left", "right", "closest"]
    distance_to_nipple: str
    prediction: str
    lesion_point_image_path: str
    lesion_projection_paths: tuple[str, str, str]
    occlusion_map_path: str


@dataclasses.dataclass
class PatientReportData:
    age: int | float | str = "N/A"

    brca: str = "N/A"
    chemo: str = "N/A"
    famrisk: str = "N/A"
    patrisk: str = "N/A"
    menostatus: str = "N/A"
    contraception: str = "N/A"

    name: str = "Unknown"
    birthdate: str = "Unknown"
    acquisition_date: str = "Unknown"

    @classmethod
    def from_dict(cls, d: dict | None):
        if d is None:
            d = {}
        else:
            d = {key.lower(): value for key, value in d.items()}

        output = {}
        # noinspection PyTypeChecker
        for field in dataclasses.fields(cls):
            if field.name not in d:
                output[field.name] = field.default
            else:
                value = d[field.name]
                if value is None:
                    value = field.default
                elif isinstance(value, (float, int)):
                    value = int(value)
                elif isinstance(value, str):
                    value = value if len(value) > 0 else field.default
                elif isinstance(value, datetime.date):
                    value = str(value)
                else:
                    raise TypeError(type(value))
                output[field.name] = value

        return PatientReportData(**output)


@dataclasses.dataclass
class ReportRenderData:
    lesions: dict[str, LesionReportData]
    patient_data: PatientReportData


def get_angle(vector_0: np.ndarray, vector_1: np.ndarray) -> float:
    """

    :param vector_0:
    :param vector_1:
    :return: Angle in rad
    """
    # noinspection PyTypeChecker
    cos_angle = np.dot(vector_0, vector_1) / (np.linalg.norm(vector_0) * np.linalg.norm(vector_1))
    return np.arccos(cos_angle)


def rotate_around_origin(vector_2d: np.ndarray, angle: float) -> np.ndarray:
    """
    :param vector_2d:
    :param angle: In rad
    :return:
    """
    if len(vector_2d) != 2:
        raise ValueError(vector_2d)

    x = vector_2d[0] * np.cos(angle) + vector_2d[1] * np.sin(angle)
    y = -vector_2d[0] * np.sin(angle) + vector_2d[1] * np.cos(angle)

    return np.asarray([x, y])


class LesionLensPDFReport(object):
    _html_template_filename = "report_template.html"
    _report_style_filename = "report_style.css"
    _lesion_loc_template_filename = "lesion_localization_template.jpg"
    _logos_filename = "all_subream_logos.png"
    _tmp_files_directory = Path("/tmp/pdf_report")

    def __init__(self,
                 patient_data: dict[str, str | int | float],
                 lesion_crops: dict[str, LesionCrop],
                 lesions_geometry: dict[str, dict[str, float]],
                 predictions: dict[str, LesionPointPrediction],
                 occlusion_maps: dict[str, torch.Tensor],
                 nipples_position: NipplesPosition | None = None,
                 templates_root: str | Path | None = None,
                 ):
        """
        :param lesion_crops: A dict mapping lesions' IDs to their crop data
        :param predictions: A dict mapping lesions' IDs to their lesion point prediction (class, probability
            and position)
        :param occlusion_maps: A dict mapping lesions' IDs to their occlusion map
        :param nipples_position:
        :param templates_root: Path to the root folder of the HTML/CSS template files. Will search CWD if not provided.
        """
        self.patient_data = patient_data

        self._check_lesion_ids(predictions, lesion_crops, occlusion_maps)

        self.lesion_ids = set(predictions.keys())
        self.lesion_crops = lesion_crops
        self.predictions = predictions
        self.lesions_geometry = lesions_geometry
        self.occlusion_maps = occlusion_maps
        self.nipples_position = nipples_position

        if templates_root is None:
            templates_root = Path.cwd()
        elif isinstance(templates_root, str):
            templates_root = Path(templates_root)

        self.templates_root: Path = templates_root
        self.html_template_path = self.find_template_path(templates_root, filename=self._html_template_filename)
        self.css_template_path = self.find_template_path(templates_root, filename=self._report_style_filename)
        self.lesion_loc_template_path = self.find_template_path(templates_root,
                                                                filename=self._lesion_loc_template_filename)
        self.logos_path = self.find_template_path(templates_root, filename=self._logos_filename)

        self.mip_dim = 0
        self.image_target_resolution = 512

    def export(self, output_path: Path) -> None:
        jinja_loader = FileSystemLoader(searchpath=self.html_template_dir)
        jinja_environment = Environment(loader=jinja_loader)
        template: Template = jinja_environment.get_template(self._html_template_filename)

        # noinspection PyTypeChecker
        shutil.copy2(self.logos_path, self.tmp_logos_path)
        render_data = dataclasses.asdict(self.get_render_data())

        html_string = template.render(**render_data)
        html = HTML(string=html_string, base_url=self.tmp_files_directory)
        css = CSS(self.css_template_path)

        html.write_pdf(output_path, stylesheets=[css])

    def get_render_data(self) -> ReportRenderData:
        lesions_data = {lesion_id: self.get_lesion_report_data(lesion_id) for lesion_id in sorted(self.lesion_ids)}
        render_data = ReportRenderData(lesions=lesions_data,
                                       patient_data=self.get_patient_report_data())
        return render_data

    def get_patient_report_data(self) -> PatientReportData:
        return PatientReportData.from_dict(self.patient_data)

    # region Lesion report data
    def get_lesion_report_data(self, lesion_id: str) -> LesionReportData:
        lesion_position = self.predictions[lesion_id].point
        lesion_point = [round(x, 1) for x in lesion_position]
        geometry = self.lesions_geometry.get(lesion_id, {})
        distance_to_nipple, closest_nipple = self.get_distance_to_closest_nipple(lesion_position)
        probability = self.predictions[lesion_id].probability * 100
        prediction = "{} ({:.1f}%)".format(self.predictions[lesion_id].name, probability)

        lesion_point_image_path, lesion_projection_paths, occlusion_map_path = self.get_lesion_images(lesion_id)

        return LesionReportData(point=lesion_point,
                                geometry=LesionReportGeometry.from_dict(geometry),
                                closest_nipple=closest_nipple,
                                distance_to_nipple=distance_to_nipple,
                                prediction=prediction,
                                lesion_point_image_path=lesion_point_image_path,
                                lesion_projection_paths=lesion_projection_paths,
                                occlusion_map_path=occlusion_map_path)

    def get_distance_to_closest_nipple(self, lesion_position: list[float]
                                       ) -> tuple[str, Literal["left", "right", "closest"]]:
        if self.nipples_position is None:
            return "Unknown", "closest"
        else:
            distance_to_nipple, closest_nipple = self.nipples_position.get_distance_to_closest_nipple(lesion_position)
            formatted_distance_to_nipple = "{:.1f}mm".format(distance_to_nipple)
            return formatted_distance_to_nipple, closest_nipple

    def get_lesion_images(self, lesion_id: str) -> tuple[str, tuple[str, str, str], str]:
        lesion_point_image_path = self.make_lesion_localization_image(lesion_id)

        lesion_projections = self.project_lesion_crop(lesion_id)
        reference_projection = lesion_projections[self.mip_dim]

        occlusion_image = self.occlusion_maps[lesion_id]
        occlusion_image = self.compute_mip(occlusion_image)
        occlusion_image = self._format_image(occlusion_image)
        occlusion_image = np.flip(occlusion_image, axis=1)
        occlusion_image = overlay_image(reference_projection, occlusion_image)

        _lesion_projection_paths = []
        for i, lesion_projection in enumerate(lesion_projections):
            lesion_crop_path = self.get_temporary_image_path(lesion_id, "lesion_crop_{}".format(i))
            _lesion_projection_paths.append(lesion_crop_path)
            cv2.imwrite(lesion_crop_path, lesion_projection)
        # noinspection PyTypeChecker
        lesion_projection_paths: tuple[str, str, str] = tuple(_lesion_projection_paths)

        occlusion_map_path = self.get_temporary_image_path(lesion_id, "occlusion_map")
        cv2.imwrite(occlusion_map_path, occlusion_image)

        return lesion_point_image_path, lesion_projection_paths, occlusion_map_path

    def make_lesion_localization_image(self, lesion_id: str) -> str:
        lesion_point_image_path = self.get_temporary_image_path(lesion_id, "lesion_point_image")

        image = cv2.imread(self.lesion_loc_template_path.as_posix())

        right_left = True
        if self.nipples_position is not None:
            left_nipple_pix_loc = np.asarray((223, 326), dtype=np.float32)
            right_nipple_pix_loc = np.asarray((428, 328), dtype=np.float32)
            if right_left:
                tmp = right_nipple_pix_loc
                right_nipple_pix_loc = left_nipple_pix_loc
                left_nipple_pix_loc = tmp
            pix_delta = right_nipple_pix_loc - left_nipple_pix_loc
            pix_x_distance = pix_delta[0]

            xz_index = np.asarray([0, 2], dtype=np.int32)
            left_nipple_position = self.nipples_position.left[xz_index]
            right_nipple_position = self.nipples_position.right[xz_index]
            position_delta = right_nipple_position - left_nipple_position
            position_x_distance = position_delta[0]

            scale = pix_x_distance / position_x_distance
            angle = get_angle(np.abs(pix_delta), np.abs(position_delta))

            lesion_position = self.lesion_crops[lesion_id].lesion_position[xz_index]
            lesion_pix_loc = (lesion_position - left_nipple_position) * scale
            lesion_pix_loc = rotate_around_origin(lesion_pix_loc, -angle)
            if right_left:
                lesion_pix_loc[1] *= -1
            lesion_pix_loc += left_nipple_pix_loc
            lesion_pix_loc = tuple([int(x) for x in lesion_pix_loc])

            radius = 8
            color = (100, 255, 50)
            thickness = 2
            image = cv2.circle(image, lesion_pix_loc, radius, color, thickness)

        cv2.imwrite(lesion_point_image_path, image)
        return lesion_point_image_path

    def project_lesion_crop(self, lesion_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lesion_crop_image = SimpleITK.GetArrayFromImage(self.lesion_crops[lesion_id].image)
        lesion_crop_image = np.expand_dims(lesion_crop_image, axis=0)
        lesion_crop_image = torch.as_tensor(lesion_crop_image)

        target_size = self.occlusion_maps[lesion_id].shape[-3:]
        resizer = Resize(spatial_size=target_size)
        lesion_crop_image = resizer(lesion_crop_image)
        lesion_crop_image = lesion_crop_image.cpu().numpy()
        lesion_crop_image = np.squeeze(lesion_crop_image)
        lesion_crop_image = np.flip(lesion_crop_image, axis=0)

        # lesion_crop_image = self.compute_mip(lesion_crop_image)
        projections = []
        for i in range(3):
            slices = tuple([slice(None) if j != i else (lesion_crop_image.shape[i] // 2)
                            for j in range(lesion_crop_image.ndim)])
            projection = lesion_crop_image[slices]
            projection = normalize(projection) * 255.0
            projection = np.expand_dims(projection, axis=-1)
            projection = self._format_image(projection)
            projections.append(projection)

        # noinspection PyTypeChecker
        return tuple(projections)

    def _format_image(self, image: np.ndarray) -> np.ndarray:
        return format_image(image,
                            target_resolution=self.image_target_resolution,
                            resize_method=cv2.INTER_NEAREST_EXACT,
                            rotate=False)

    def compute_mip(self, image_3d: np.ndarray | torch.Tensor | MetaTensor) -> np.ndarray:
        if isinstance(image_3d, torch.Tensor):
            image_3d = image_3d.cpu().numpy()

        image_3d = np.squeeze(image_3d)
        image_2d: np.ndarray = image_3d.max(axis=self.mip_dim)
        image_2d = normalize(image_2d) * 255.0
        image_2d = np.expand_dims(image_2d, axis=-1)

        return image_2d

    def get_temporary_image_path(self,
                                 lesion_id: str,
                                 image_name: str
                                 ) -> str:
        return (self.tmp_files_directory / "{}_{}.png".format(lesion_id, image_name)).as_posix()

    # endregion

    @property
    def html_template_dir(self) -> Path:
        return self.html_template_path.parent

    @property
    def tmp_files_directory(self) -> Path:
        if not self._tmp_files_directory.exists():
            self._tmp_files_directory.mkdir(parents=True)

        return self._tmp_files_directory

    @property
    def tmp_logos_path(self) -> Path:
        return self.tmp_files_directory / "all_subream_logos.png"

    @staticmethod
    def parse_patient_information_field(value: str) -> str:
        if value is None:
            return ""

        if isinstance(value, int):
            return str(value)

        if not isinstance(value, str):
            raise ValueError("Excepted value to be a str, int or None, got {}".format(type(value)))

        if len(value) == 0:
            return ""

        value = value.replace("_", " ")
        value = value.capitalize()

        return value

    @staticmethod
    def find_template_path(root: Path, filename: str) -> Path:
        path = find_first_path(root, pattern=filename, recursive=True)
        if path is None:
            raise FileNotFoundError("Could not find template `{}` in `{}`.".format(filename, root))

        return path

    @staticmethod
    def _check_lesion_ids(predictions: dict[str, LesionPointPrediction],
                          lesion_crops: dict[str, LesionCrop],
                          occlusion_maps: dict[str, torch.Tensor]) -> None:
        predictions_ids = set(predictions)
        lesion_crops_ids = set(lesion_crops)
        occlusion_maps_ids = set(occlusion_maps)

        if (predictions_ids != lesion_crops_ids) or (lesion_crops_ids != occlusion_maps_ids):
            raise ValueError("Lesion IDs do not match: "
                             "`predictions_ids`: {}, "
                             "`lesion_crops_ids`: {} and "
                             "`occlusion_maps_ids`: {}.".format(predictions_ids,
                                                                lesion_crops_ids,
                                                                occlusion_maps_ids))

# endregion
