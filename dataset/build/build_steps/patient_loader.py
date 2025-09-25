import SimpleITK
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import os
from typing import Iterator

from mindful_core.utils.parsing import parse_last_number

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger


class SubreamPatientLoader(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamPatientLoader, self).__init__(build_data, output_root, logger)

    def run(self) -> None:
        self.find_all_patient_images()
        if self.data.use_sub:
            self.load_and_subtract_images()

        self.find_all_patient_masks()
        self.aggregate_all_patient_masks()

        self.list_all_masks_roi_points_filepaths()
        self.load_all_masks_roi_points()

    def clean_up(self) -> None:
        self.remove_temporary_folder()

    # region Find Images
    def find_all_patient_images(self) -> None:
        patient_folders = self.data.get_patient_folders()
        # For debug/dev purposes
        if (self.data.max_patients is not None) and (self.data.max_patients > 0):
            patient_folders = {key: patient_folders[key] for key in
                               list(patient_folders.keys())[:self.data.max_patients]}
        patients_images_paths: dict[str, list[Path]] = {}

        max_patients = self.data.max_patients
        for i, patient_id in enumerate(patient_folders):
            patient_folder = patient_folders[patient_id]
            patient_phases_paths = self.find_patient_base_phases(patient_folder)
            patients_images_paths[patient_id] = patient_phases_paths
            if (max_patients is not None) and ((i + 1) >= max_patients):
                break

        self.data.patients_ids = list(patients_images_paths.keys())
        self.data.patients_images_paths = patients_images_paths

    @staticmethod
    def find_patient_base_phases(patient_folder: Path) -> list[Path]:
        patient_phases_paths: dict[int, Path] = {}
        for path in patient_folder.glob(pattern="*.mha"):
            *_, phase_info = path.stem.split("_")
            # phase info expected to match `t{phase_id}`
            phase_id = phase_info[1:]
            if not phase_id.isdigit():
                continue

            patient_phases_paths[int(phase_id)] = path

        sorted_patient_phases_paths = [patient_phases_paths[file_id] for file_id in sorted(patient_phases_paths)]
        return sorted_patient_phases_paths

    # endregion

    # region Subtract images
    def load_and_subtract_images(self) -> None:
        if "subtracted_images_paths" in self.data.resume_from:
            subtracted_images_paths = self.discover_subtracted_images_from_previous_build()
        else:
            subtracted_images_paths: dict[str, list[Path]] = {}

        progress_bar = tqdm(self.data.patients_ids, desc="Subtracting patient images...")
        incompatible_phases: dict[Path, RuntimeError] = {}
        for patient_id in progress_bar:
            if patient_id in subtracted_images_paths:
                continue
            progress_bar.set_description("Subtracting patient images... (id: {})".format(patient_id))

            patient_folder = "Patient_{}".format(patient_id)
            patient_subtracted_images_folder = self.get_images_output_path("subtracted") / patient_folder
            patient_subtracted_images_folder.mkdir(parents=True)
            try:
                patient_phases = self.data.read_patient_images(patient_id, subtracted=False)
                first_phase = patient_phases.pop(0)

                subtracted_mins, subtracted_maxs = [], []
                subtracted_phases = []
                for i, phase in enumerate(patient_phases):
                    try:
                        phase.CopyInformation(first_phase)
                    except RuntimeError as error:
                        phase_path = self.data.patients_images_paths[patient_id][i + 1]
                        incompatible_phases[phase_path] = error
                        continue

                    subtracted: SimpleITK.Image = phase - first_phase
                    subtracted_min, subtracted_max = SimpleITK.MinimumMaximum(subtracted)

                    subtracted_phases.append(subtracted)
                    subtracted_mins.append(subtracted_min)
                    subtracted_maxs.append(subtracted_max)

                if len(subtracted_phases) == 0:
                    continue

                phases_min, phases_max = min(subtracted_mins), max(subtracted_maxs)
                output_paths: list[Path] = []
                for i, subtracted in enumerate(subtracted_phases):
                    if self.data.normalize_subtracted:
                        subtracted = (subtracted - phases_min) / (phases_max - phases_min)

                    output_path = patient_subtracted_images_folder / "phase_{}.mha".format(i)
                    SimpleITK.WriteImage(subtracted, output_path.as_posix())
                    output_paths.append(output_path)

                subtracted_images_paths[patient_id] = output_paths

            except KeyboardInterrupt:
                break

        if len(incompatible_phases) > 0:
            errors = ["{}: {}".format(phase_path, phase_error)
                      for phase_path, phase_error in incompatible_phases.items()]
            error_list = "\r\n".join(errors)
            error_message = "Detected {} phase(s) incompatible with their first phase " \
                            "during the subtraction process: \r\n{}".format(len(incompatible_phases), error_list)
            self.logger.warn(error_message, RuntimeError)

        self.data.subtracted_images_paths = subtracted_images_paths

    def discover_subtracted_images_from_previous_build(self) -> dict[str, list[Path]]:
        previous_subtracted_images_root = Path(self.data.resume_from["subtracted_images_paths"])

        prefix_length = len("phase_")

        def to_phase_index(path: Path):
            return int(path.stem[prefix_length:])

        subtracted_images_paths: dict[str, list[Path]] = {}
        progress_bar = tqdm(self.data.patients_ids, desc="Discovering subtracted images from previous build...")
        for patient_id in progress_bar:
            patient_folder = "Patient_{}".format(patient_id)
            patient_subtracted_images_folder = previous_subtracted_images_root / patient_folder
            if patient_subtracted_images_folder.exists():
                description = "Discovering subtracted images from previous build... (id: {})".format(patient_id)
                progress_bar.set_description(description)

                output_paths = [filepath for filepath in patient_subtracted_images_folder.glob("*phase_*.mha")]
                output_paths = list(sorted(output_paths, key=to_phase_index))
                subtracted_images_paths[patient_id] = output_paths

        return subtracted_images_paths

    def remove_temporary_folder(self) -> None:
        path = self.get_images_output_path("subtracted")
        if path.exists():
            shutil.rmtree(path)
            self.log("PatientLoader - Removed temporary `subtracted` image folder.")

    # endregion

    # region Masks
    def find_all_patient_masks(self) -> None:
        masks_roi_filepaths: dict[str, Path] = {}
        for filepath in self.data.masks_folder.glob("*.mha"):
            patient_id, _, _, lesion_index, _ = filepath.stem.split("_")
            patient_id = parse_last_number(patient_id)
            lesion_id = "{}_{}".format(patient_id, lesion_index)
            masks_roi_filepaths[lesion_id] = filepath

        self.data.masks_filepaths = masks_roi_filepaths

    def aggregate_all_patient_masks(self) -> None:
        output_dir = self.get_masks_output_path("aggregated")
        filepaths: dict[str, Path] = {}

        progress_bar = tqdm(self.data.patients_ids, desc="Aggregating masks...")
        incompatible_masks: dict[str, RuntimeError] = {}
        for patient_id in progress_bar:
            try:
                patient_mask = self.aggregate_patient_masks(patient_id)
            except RuntimeError as error:
                incompatible_masks[patient_id] = error
                continue

            if patient_mask is None:
                continue

            filepath = output_dir / "{}.mha".format(patient_id)
            SimpleITK.WriteImage(patient_mask, filepath.as_posix())
            filepaths[patient_id] = filepath

        if len(incompatible_masks) > 0:
            errors = ["{}: {}".format(phase_path, phase_error)
                      for phase_path, phase_error in incompatible_masks.items()]
            error_list = "\r\n".join(errors)
            error_message = "Detected {} patient(s) with masks incompatible with each others " \
                            "during the subtraction process: \r\n{}".format(len(incompatible_masks), error_list)
            self.logger.warn(error_message, RuntimeError)

        self.data.aggregated_masks_filepaths = filepaths

    def aggregate_patient_masks(self, patient_id: str) -> SimpleITK.Image | None:
        masks_filepaths = self.data.masks_filepaths
        patient_id_ = patient_id + "_"
        patient_lesion_ids = [lesion_id for lesion_id in masks_filepaths.keys()
                              if lesion_id.startswith(patient_id_)]
        lesion_labels = self.data.get_lesion_labels_by_id()

        output = None
        for lesion_id in patient_lesion_ids:
            mask_filepath = masks_filepaths[lesion_id]
            mask = SimpleITK.ReadImage(mask_filepath.as_posix())
            lesion_label = lesion_labels[lesion_id]
            mask_value = self.data.get_lesion_label_value(lesion_label)
            # noinspection PyTypeChecker
            mask: SimpleITK.Image = (mask > 0) * mask_value

            if output is None:
                output = mask
            else:
                # Assumes ROIs do not overlap !
                mask.CopyInformation(output)
                output |= mask

        if output is None:
            return None

        array = SimpleITK.GetArrayFromImage(output)
        if array.max() >= 4:
            raise RuntimeError("Invalid label value detected for {} !".format(patient_id))

        return output

    # endregion

    # region Load Points
    def list_all_masks_roi_points_filepaths(self):
        known_lesion_ids = self.data.get_lesion_labels_by_id()
        valid_filepaths: dict[str, Path] = {}
        wrong_labels: list[Path] = []
        unknown_lesions: list[Path] = []

        for filepath in self.data.masks_roi_points_folder.iterdir():
            if filepath.match("*_lesion_*_*.txt"):
                lesion_id = self.data.get_mask_lesion_id(filepath)
                lesion_label = self.data.get_lesion_label(filepath)

                # noinspection PyPackages
                if lesion_id not in known_lesion_ids:
                    unknown_lesions.append(filepath)
                elif lesion_label != known_lesion_ids[lesion_id]:
                    wrong_labels.append(filepath)
                else:
                    valid_filepaths[lesion_id] = filepath

        self._log_masks_roi_points_status(wrong_labels, unknown_lesions)
        # noinspection PyTypeChecker
        self._backup_masks_roi_points_files(valid_filepaths.values())

        self.data.masks_roi_points_filepaths = valid_filepaths

    def _backup_masks_roi_points_files(self, filepaths: Iterator[Path]) -> None:
        points_folder = self.get_output_path("points")
        for filepath in filepaths:
            shutil.copy2(filepath, points_folder / filepath.name)

    def _log_masks_roi_points_status(self, wrong_labels: list[Path], unknown_lesions: list[Path]) -> None:
        if len(wrong_labels) > 0:
            message = ("," + os.linesep).join(["\t" + path.as_posix() for path in wrong_labels])
            self.warn("{} points with wrong labels:{}{}".format(len(wrong_labels), os.linesep, message))
        else:
            self.log("SUCCESS: No wrong labels in ROI points.")

        if len(unknown_lesions) > 0:
            message = ("," + os.linesep).join(["\t" + path.as_posix() for path in unknown_lesions])
            self.warn("{} unknown lesions:{}{}".format(len(unknown_lesions), os.linesep, message))
        else:
            self.log("SUCCESS: No unknown samples in ROI points.")

    def load_all_masks_roi_points(self):
        masks_roi_points: dict[str, np.ndarray] = {}
        for lesion_id, points_filepath in self.data.masks_roi_points_filepaths.items():
            with open(points_filepath, "r") as points_file:
                lines = [line.rstrip() for line in points_file]
                if len(lines) < 2:
                    raise RuntimeError("Points file is empty ({})".format(points_filepath.as_posix()))
                points_count = int(lines.pop(0))
                points = [np.asarray(line.split(" "), dtype=np.float32) for _, line in zip(range(points_count), lines)]
                masks_roi_points[lesion_id] = np.stack(points, axis=0)

        self.data.masks_roi_points = masks_roi_points

    # endregion
