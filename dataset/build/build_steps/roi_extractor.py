import SimpleITK
import numpy as np
from pathlib import Path
from tqdm import tqdm

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.roi_extraction import save_rois


class SubreamROIExtractor(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamROIExtractor, self).__init__(build_data, output_root, logger)

    def run(self) -> None:
        self.extract_all_rois()
        self.save_rois()

    # region Extraction

    def extract_all_rois(self):
        progress_bar = tqdm(self.data.patients_ids, desc="Extracting ROIs...")
        patients_points = self.as_patient_lesion_dict(self.data.masks_roi_points)

        extracted_rois: dict[str, list[SimpleITK.Image]] = {}
        extracted_masks: dict[str, list[SimpleITK.Image]] = {}
        patients_with_no_matching_points_file: list[str] = []
        for patient_id in progress_bar:
            try:
                if patient_id not in patients_points:
                    patients_with_no_matching_points_file.append(patient_id)
                    continue

                if not self.data.has_patient_images(patient_id):
                    continue

                progress_bar.set_description("Extracting ROIs... ({})".format(patient_id))
                patient_images = self.data.read_patient_images(patient_id)
                patient_mask = self.data.get_aggregated_patient_mask(patient_id)

                for lesion_index, lesion_points in patients_points[patient_id].items():
                    lesion_id = "{}_{}".format(patient_id, lesion_index)

                    for point_index, point in enumerate(lesion_points):
                        point: np.ndarray
                        roi_id = "{}_{}".format(lesion_id, point_index)
                        extracted_rois[roi_id] = self.extract_roi(patient_images, point, self.data.roi_size)
                        if patient_mask is not None:
                            extracted_masks[roi_id] = self.extract_roi([patient_mask], point, self.data.roi_size, 
                                                                       use_nearest=True)

            except KeyboardInterrupt:
                # Catching KeyboardInterrupt to stop the process earlier
                break

        self._log_patients_matching_points(patients_with_no_matching_points_file)
        self.data.discard_patients_ids(patients_with_no_matching_points_file)
        self.data.extracted_rois = extracted_rois
        self.data.extracted_agg_masks = extracted_masks

    @staticmethod
    def extract_patient_rois(patient_id: str,
                             patient_images: list[SimpleITK.Image],
                             patient_masks_roi_points: dict[str, np.ndarray],
                             roi_size: float,
                             ) -> dict[str, list[SimpleITK.Image]]:
        extracted_rois: dict[str, list[SimpleITK.Image]] = {}

        for lesion_index, lesion_points in patient_masks_roi_points.items():
            lesion_id = "{}_{}".format(patient_id, lesion_index)
            for point_index, point in enumerate(lesion_points):
                point: np.ndarray
                roi_id = "{}_{}".format(lesion_id, point_index)
                extracted_rois[roi_id] = SubreamROIExtractor.extract_roi(patient_images, point, roi_size)

        return extracted_rois

    @staticmethod
    def extract_roi(images: list[SimpleITK.Image],
                    point: np.ndarray,
                    roi_size: float | np.ndarray,
                    use_nearest=False
                    ) -> list[SimpleITK.Image]:
        if not isinstance(point, np.ndarray):
            raise ValueError("Expected point to be an np.ndarray, got a `{}`.".format(type(point)))

        if point.shape != (3,):
            raise ValueError("Expected point to be a vector with length 3 (x, y, z), got an array with shape {}."
                             .format(point.shape))

        identity = SimpleITK.Transform(3, SimpleITK.sitkIdentity)
        output_direction = np.reshape(np.eye(3, 3), newshape=[9])

        output_images = []
        for image in images:
            # output_size: the output size of the voxel array (int)
            # spacing: the size of voxels (in mm, float)
            # output_origin: the origin of the output image (in mm, float)
            spacing = np.asarray(image.GetSpacing(), dtype=point.dtype)
            output_size = roi_size / spacing + 0.5
            output_origin = point - roi_size * 0.5

            # region Convert np array to built-in types for SimpleITK
            output_size = [int(_x) for _x in output_size]
            spacing = [float(_x) for _x in spacing]
            output_origin = [float(_x) for _x in output_origin]
            # endregion
            interpolator = SimpleITK.sitkNearestNeighbor if use_nearest else SimpleITK.sitkLinear
            output = SimpleITK.Resample(image, output_size, identity, interpolator,
                                        output_origin, spacing, output_direction)
            output_images.append(output)
        return output_images

    def _log_patients_matching_points(self, patients_with_no_matching_points_file: list[str]):
        missing_count = len(patients_with_no_matching_points_file)
        if missing_count > 0:
            message = ", ".join(patients_with_no_matching_points_file)
            self.warn("{} patient images with no points: {}".format(missing_count, message))
        else:
            self.log("SUCCESS: No known patients were missing images.")

    # endregion

    # region Saving
    def save_rois(self):
        extracted_roi_paths = save_rois(output_images_folder=self.get_images_output_path("default"),
                                        rois=self.data.extracted_rois,
                                        short_labels=self.data.get_lesion_labels_by_id(),
                                        image_names="phase")
        self.data.extracted_roi_paths = extracted_roi_paths

        extracted_masks_paths = save_rois(output_images_folder=self.get_masks_output_path("roi"),
                                          rois=self.data.extracted_agg_masks,
                                          short_labels=self.data.get_lesion_labels_by_id(),
                                          image_names="mask")
        self.data.extracted_agg_masks_paths = extracted_masks_paths

    # endregion
