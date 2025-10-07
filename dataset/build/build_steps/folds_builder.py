import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
import os
import warnings

from mindful_core.utils.data_constants import SCAN_ID, SUBSET_ID, LABEL, DEFAULT_IMAGE_COLUMN, DEFAULT_MASK_COLUMN

from mindful_subream.dataset.build.build_steps.build_step import SubreamBuildStep
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger


class SubreamSample(object):
    def __init__(self,
                 lesion_id: str,
                 label: str | int,
                 image_path: str | Path,
                 point_index: int,
                 mask_path: str | Path | None = None,
                 ) -> None:
        self.lesion_id = lesion_id
        self.label: str = str(label)
        self.image_path = image_path
        self.point_index = point_index
        self.mask_path = mask_path

    @property
    def scan_id(self) -> str:
        if self.is_augmentation:
            return "{}_{}_aug_{}".format(self.lesion_id, self.point_index, self.augmentation_index)
        else:
            return "{}_{}".format(self.lesion_id, self.point_index)

    @property
    def is_augmentation(self) -> bool:
        return "aug" in self.image_name

    @property
    def augmentation_index(self) -> int | None:
        if self.is_augmentation:
            return int(self.image_name.split("_")[-1].replace(".mha", ""))
        else:
            return None

    @property
    def image_name(self) -> str:
        return os.path.basename(self.image_path)


class SubreamPatient(object):
    def __init__(self, patient_id: str, samples: list[SubreamSample] = None):
        self.patient_id = patient_id
        self._samples: dict[str, SubreamSample] = {}

        if samples is not None:
            self.add_samples(samples)

    def get_label_counts(self) -> dict[str | int, int]:
        label_counts = {}
        for sample in self.get_samples():
            if sample.label not in label_counts:
                label_counts[sample.label] = 0
            label_counts[sample.label] += 1
        return label_counts

    def get_instance_count(self, classes: list[str]) -> dict[str, float]:
        prevalence = {cls: 0 for cls in classes}
        for sample in self.get_samples():
            prevalence[sample.label] += 1
        return prevalence

    def add_sample(self, sample: SubreamSample):
        if sample.scan_id in self._samples:
            warnings.warn("Found duplicate sample scan `{}` for "
                          "patient `{}`, skipping...".format(sample.scan_id, self.patient_id))
        else:
            self._samples[sample.scan_id] = sample

    def add_samples(self, samples: list[SubreamSample]):
        for sample in samples:
            self.add_sample(sample)

    def get_samples(self) -> list[SubreamSample]:
        return list(self._samples.values())

    def filter_samples(self, scan_ids: list[str]) -> bool:
        self._samples = {scan_id: self._samples[scan_id] for scan_id in scan_ids if scan_id in self._samples}
        return len(self._samples) > 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)


def get_patients(root: str) -> list[SubreamPatient]:
    patients: dict[str, SubreamPatient] = {}

    for folder, _, filenames in os.walk(root):
        # folder = os.path.relpath(folder, root)
        subtract_only = any([("sub.mha" in filename) for filename in filenames])
        for filename in filenames:
            if subtract_only and "sub.mha" not in filename:
                continue

            if filename.endswith(".mha"):
                sample_info = filename.split("_")
                patient_id = sample_info[1] if subtract_only else sample_info[0]
                scan_index = sample_info[2]
                point_index = int(sample_info[3])

                lesion_id = "{}_{}".format(patient_id, scan_index)
                if "aug" not in sample_info:
                    label = sample_info[-1][:-4]
                else:
                    aug_idx = sample_info.index("aug")
                    label = sample_info[aug_idx - 1]

                filepath = os.path.join(folder, filename)

                if patient_id not in patients:
                    patients[patient_id] = SubreamPatient(patient_id)

                sample = SubreamSample(lesion_id, label, filepath, point_index)
                patients[patient_id].add_sample(sample)

    return list(patients.values())


labels_mode_map = {
    "b_g_m": {"b": 0, "g": 1, "m": 2},
    "bg_m": {"b": 0, "m": 1, "g": 0},
    "b_m": {"b": 0, "m": 1},
    "g_m": {"g": 0, "m": 1},
    "bm_g": {"b": 0, "m": 0, "g": 1},
    "b_g": {"b": 0, "g": 1},
}


def load_labels(labels_file: str, mode: str) -> dict[str, str]:
    labels_data_frame = pd.read_csv(labels_file, encoding="latin1")
    valid_labels_filter = labels_data_frame["EXPORT"] == "OK"
    labels_data_frame = labels_data_frame[valid_labels_filter]
    labels_array = labels_data_frame.values

    patient_code_index = labels_data_frame.columns.get_loc("Code")
    lesion_code_index = labels_data_frame.columns.get_loc("n° lésion")
    label_index = labels_data_frame.columns.get_loc("type")

    labels = {}
    for entry in labels_array:
        patient_code = entry[patient_code_index]
        lesion_code = entry[lesion_code_index]
        label = entry[label_index].lower()
        if label in mode:
            # noinspection PyUnresolvedReferences
            label = str(labels_mode_map[mode][label])
            sample_code = "{}_{}".format(int(patient_code), int(lesion_code))
            labels[sample_code] = label.lower()

    return labels


def balance_lymphnodes(patients: list[SubreamPatient], lymphnode_label="g") -> list[SubreamPatient]:
    all_samples = sum([patient.get_samples() for patient in patients], [])

    counts_per_label: dict[str, int] = {}
    samples_per_label: dict[str, list[SubreamSample]] = {}
    for sample in all_samples:
        label = sample.label.lower() if isinstance(sample.label, str) else sample.label
        if label not in counts_per_label:
            counts_per_label[label] = 0
            samples_per_label[label] = []
        counts_per_label[label] += 1
        samples_per_label[label].append(sample)

    lymphnodes: list[SubreamSample] = samples_per_label[lymphnode_label]
    indices = np.arange(counts_per_label["g"], dtype=np.int32)
    np.random.shuffle(indices)

    max_lymphnodes = max([count for label, count in
                          counts_per_label.items() if label != lymphnode_label])
    indices = indices[:max_lymphnodes]
    lymphnodes: list[SubreamSample] = [lymphnodes[i] for i in indices]

    samples = sum([samples for label, samples in samples_per_label.items()
                   if label != lymphnode_label], lymphnodes)
    samples_ids = [sample.scan_id for sample in samples]
    filtered_patients = []
    for patient in patients:
        not_empty = patient.filter_samples(samples_ids)
        if not_empty:
            filtered_patients.append(patient)

    return filtered_patients


def update_labels(patients: list[SubreamPatient], labels: dict[str, str]) -> list[SubreamPatient]:
    updated_patients = []

    for patient in patients:
        updated_samples = []
        for sample in patient.get_samples():
            if sample.lesion_id in labels:
                updated_samples.append(SubreamSample(sample.lesion_id, labels[sample.lesion_id],
                                                     sample.image_path, sample.point_index, sample.mask_path))

        if len(updated_samples) > 0:
            updated_patients.append(SubreamPatient(patient.patient_id, updated_samples))

    return updated_patients


def make_random_folds(patients: list[SubreamPatient], folds_count: int) -> list[list[SubreamPatient]]:
    patients = copy.copy(patients)
    folds = [[] for _ in range(folds_count)]

    def get_smallest_fold_index() -> int:
        return int(np.argmin([len(fold) for fold in folds]))

    while len(patients) > 0:
        random_index = np.random.randint(len(patients))
        patient = patients.pop(random_index)
        fold_index = get_smallest_fold_index()
        folds[fold_index].append(patient)

    return folds


def get_folds_instance_count(folds: list[list[SubreamPatient]], classes: list[str]) -> list[list[int]]:
    folds_instance_count = []
    for i, fold in enumerate(folds):
        fold_instance_count = [0 for _ in classes]
        for patient in fold:
            patient_instance_count = patient.get_instance_count(classes).values()
            fold_instance_count = [x + y for x, y in zip(fold_instance_count, patient_instance_count)]
        folds_instance_count.append(fold_instance_count)

    return folds_instance_count


def evaluate_folds(folds: list[list[SubreamPatient]], classes: list[str], target_instance_count: list[float]) -> float:
    target_instance_count = np.asarray(target_instance_count, dtype=np.float32)

    folds_instance_count = get_folds_instance_count(folds, classes)
    folds_instance_count = np.asarray(folds_instance_count, dtype=np.float32)

    distances = [np.sum(np.square(target_instance_count - fold_instance_count))
                 for fold_instance_count in folds_instance_count]
    max_distance = float(max(distances))

    return max_distance


def stochastic_folds_generation(patients: list[SubreamPatient],
                                folds_count: int,
                                iterations: int
                                ) -> list[list[SubreamPatient]]:
    classes = []
    sample_count = 0
    for patient in patients:
        sample_count += patient.sample_count
        for sample in patient.get_samples():
            if sample.label not in classes:
                classes.append(sample.label)

    target_instance_count = {cls: 0 for cls in classes}
    for patient in patients:
        patient_prevalence = patient.get_instance_count(classes)
        for cls in patient_prevalence:
            target_instance_count[cls] += patient_prevalence[cls]

    for cls in target_instance_count:
        target_instance_count[cls] /= folds_count
    target_instance_count = list(target_instance_count.values())

    best_distance: float | None = None
    best_folds = None
    for _ in tqdm(range(iterations), desc="Generating random folds"):
        folds = make_random_folds(patients, folds_count)
        distance = evaluate_folds(folds, classes, target_instance_count)
        if (best_distance is None) or (best_distance > distance):
            best_distance = distance
            best_folds = folds

    return best_folds


def patient_to_samples_folds(patient_folds: list[list[SubreamPatient]]) -> list[list[SubreamSample]]:
    return [sum([patient.get_samples() for patient in patient_fold], [])
            for patient_fold in patient_folds]


def make_cross_validation_folds(partitions: list[list[SubreamSample]],
                                test_partitions_count: int,
                                validation_partitions_count: int
                                ) -> list[pd.DataFrame]:
    """
        Order: Test > Validation > Train
    """
    folds_count = len(partitions)
    data_frames = []
    for start_index in range(folds_count):
        indices = [(start_index + i) % folds_count for i in range(folds_count)]

        test_indices = indices[:test_partitions_count]
        validation_indices = indices[test_partitions_count:test_partitions_count + validation_partitions_count]
        train_indices = indices[test_partitions_count + validation_partitions_count:]

        test_partitions = sum([partitions[i] for i in test_indices], [])
        validation_partitions = sum([partitions[i] for i in validation_indices], [])
        train_partitions = sum([partitions[i] for i in train_indices], [])

        fold = {
            "test": test_partitions,
            "validation": validation_partitions,
            "train": train_partitions,
        }

        table = []
        for subset_name, samples in fold.items():
            for sample in samples:
                if (subset_name == "test") and sample.is_augmentation:
                    continue

                line = [sample.scan_id, subset_name, sample.label, sample.image_path, sample.mask_path]
                table.append(line)
        
        data_frame = pd.DataFrame(table, columns=[SCAN_ID, SUBSET_ID, LABEL, DEFAULT_IMAGE_COLUMN, DEFAULT_MASK_COLUMN])
        data_frames.append(data_frame)
    return data_frames


def save_cross_validation_folds(folds: list[pd.DataFrame],
                                output_folder: str | Path
                                ) -> list[Path]:
    output_filepaths = []

    for i, data_frame in enumerate(folds):
        output_filepath = Path(output_folder, "subream_fold_{:02d}.csv".format(i))
        data_frame.to_csv(output_filepath, index=False)
        output_filepaths.append(output_filepath)

    return output_filepaths


def get_fold_version(balanced_lymphnodes: bool) -> str:
    return "balanced_lymphnodes" if balanced_lymphnodes else "default"


class SubreamFoldsBuilder(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamFoldsBuilder, self).__init__(build_data, output_root, logger)

        # fold version > mode > patient id => partition id
        self.missing_patient_ids_map: dict[str, dict[str, dict[str, int]]] | None = None

    def run(self) -> None:
        for balanced_lymphnodes in [True, False]:
            for mode in labels_mode_map:
                folds = self.generate_folds_for_mode(self.data.extracted_roi_paths,
                                                     self.data.extracted_agg_masks_paths,
                                                     mode,
                                                     balanced_lymphnodes)

                fold_version = get_fold_version(balanced_lymphnodes)
                current_folds_folder = self.get_scan_folds_path() / fold_version / mode
                current_folds_folder.mkdir(parents=True)

                folds_paths = save_cross_validation_folds(folds, current_folds_folder)

                self.register_folds("scan", fold_version, mode, folds_paths)

        if self.data.extracted_mip_paths is not None:
            self.convert_folds(target="mip")

        if self.data.extracted_breasts_images_paths is not None:
            self.assign_missing_patient_ids()
            self.convert_folds(target="breasts_bounding_boxes")

    def build_folds_for_mode(self,
                             mode: str,
                             balanced_lymphnodes: bool
                             ) -> list[pd.DataFrame]:
        reference_dir = None
        if self.data.reference_folds is not None:
            reference_dir = self.data.reference_folds
            if reference_dir.name != "folds":
                reference_dir = reference_dir / "folds"

            reference_dir = reference_dir / "scan" / get_fold_version(balanced_lymphnodes) / mode

            if not reference_dir.exists():
                reference_dir = None

        if reference_dir is None:
            folds = self.generate_folds_for_mode(self.data.extracted_roi_paths,
                                                 self.data.extracted_agg_masks_paths,
                                                 mode,
                                                 balanced_lymphnodes)
        else:
            folds = self.build_folds_from_reference(reference_dir)

        return folds

    def build_folds_from_reference(self, reference_dir: Path) -> list[pd.DataFrame]:
        reference_paths = list(reference_dir.glob("*.csv"))
        reference_paths = sorted(reference_paths, key=lambda path: int(path.stem.split("_")[-1]))

        folds = []
        for reference_path in reference_paths:
            fold = pd.read_csv(reference_path, index_col=SCAN_ID)

            # region Init update
            update = {DEFAULT_IMAGE_COLUMN: {}, DEFAULT_MASK_COLUMN: {}}

            # endregion

            # region Empty previous data first
            for column_name in update:
                fold[column_name] = ""

            # endregion

            # region Create update
            for point_id in self.data.extracted_roi_paths:
                rois_filepaths = self.data.extracted_roi_paths[point_id]
                extracted_masks_paths = self.data.extracted_agg_masks_paths[point_id]

                last_phase_path = rois_filepaths[-1].as_posix()
                mask_path = extracted_masks_paths[0].as_posix()

                update[DEFAULT_IMAGE_COLUMN][point_id] = last_phase_path
                update[DEFAULT_MASK_COLUMN][point_id] = mask_path

            # endregion

            # region Apply update
            for column_name, column_values in update.items():
                fold[column_name] = column_values

            # endregion

            folds.append(fold)

        return folds

    def generate_folds_for_mode(self,
                                extracted_images_paths: dict[str, list[Path]],
                                extracted_masks_paths: dict[str, list[Path]],
                                mode: str,
                                balanced_lymphnodes: bool,
                                ) -> list[pd.DataFrame]:
        self.log("Generating folds for mode `{}` (balance lymphnodes = {})".format(mode, balanced_lymphnodes))

        indexed_patients: dict[str, SubreamPatient] = {}
        lesion_labels = self.data.get_lesion_labels_by_id()

        for point_id, rois_filepaths in extracted_images_paths.items():
            patient_id, lesion_index, point_index = point_id.split("_")
            point_index = int(point_index)
            lesion_id = "{}_{}".format(patient_id, lesion_index)

            if patient_id not in indexed_patients:
                indexed_patients[patient_id] = SubreamPatient(patient_id)

            label = lesion_labels[lesion_id]
            if len(rois_filepaths) == 0:
                raise RuntimeError("Empty ROIs filepaths for patient `{}` (and lesion index `{}`)."
                                   .format(patient_id, lesion_index))
            last_phase_path = rois_filepaths[-1].as_posix()
            mask_path = extracted_masks_paths[point_id][0]
            sample = SubreamSample(lesion_id, label, last_phase_path, point_index, mask_path)
            indexed_patients[patient_id].add_sample(sample)

        patients = list(indexed_patients.values())
        if balanced_lymphnodes:
            patients = balance_lymphnodes(patients)

        patients = self.map_labels(patients, mode)

        patient_folds = stochastic_folds_generation(patients,
                                                    self.data.folds_count,
                                                    self.data.stochastic_folds_iterations)
        sample_folds = patient_to_samples_folds(patient_folds)

        folds = make_cross_validation_folds(sample_folds,
                                            self.data.test_folds_count,
                                            self.data.validation_folds_count)
        return folds

    @staticmethod
    def map_labels(patients: list[SubreamPatient], mode: str) -> list[SubreamPatient]:
        current_labels_mode_map = labels_mode_map[mode]
        updated_patients: list[SubreamPatient] = []
        for patient in patients:
            kept_samples_ids: list[str] = []
            for sample in patient.get_samples():
                sample_label = sample.label.lower()
                if sample_label in current_labels_mode_map:
                    sample.label = str(current_labels_mode_map[sample_label])
                    kept_samples_ids.append(sample.scan_id)
            keep_patient = patient.filter_samples(kept_samples_ids)
            if keep_patient:
                updated_patients.append(patient)

        return updated_patients

    def convert_folds(self, target: str) -> None:
        original_folds = self.data.folds["scan"]

        for fold_version in original_folds:
            for mode in original_folds[fold_version]:
                target_folder_path = self.get_folds_output_path(target) / fold_version / mode
                target_folder_path.mkdir(parents=True, exist_ok=True)

                target_folds = []
                for i, original_fold_path in enumerate(original_folds[fold_version][mode]):
                    original_fold = pd.read_csv(original_fold_path, index_col=SCAN_ID)
                    target_fold = self.convert_fold(original_fold, target, fold_version, mode, fold_number=i)
                    target_fold_path = target_folder_path / original_fold_path.name
                    target_fold.to_csv(target_fold_path)
                    target_folds.append(target_fold_path)
                self.register_folds(target, fold_version, mode, target_folds)

    def convert_fold(self,
                     original_fold: pd.DataFrame,
                     target: str,
                     fold_version: str,
                     mode: str,
                     fold_number: int,
                     ) -> pd.DataFrame:
        if target == "mip":
            return self.convert_fold_for_mip(original_fold)
        elif target == "breasts_bounding_boxes":
            # TODO: when a patient with bbox isn't in the reference fold, 
            #   first add them in a partition (and make it consistent across folds)
            return self.convert_fold_for_breasts_bounding_boxes(original_fold, fold_version, mode, fold_number)
        else:
            raise ValueError(target)

    def convert_fold_for_mip(self, original_fold: pd.DataFrame) -> pd.DataFrame:
        target_data = {}

        for mip_id, mip_path in self.data.extracted_mip_paths.items():
            patient_id, lesion_index, point_index, version_index = mip_id.split("_")
            point_id = "{}_{}_{}".format(patient_id, lesion_index, point_index)
            version_id = "{}_{}".format(point_id, version_index)

            if point_id not in original_fold.index:
                break

            row = original_fold.loc[point_id].copy()
            row["image:mip"] = mip_path
            target_data[version_id] = row

        columns = original_fold.columns + ["image:mip"]
        target_fold = pd.DataFrame.from_dict(data=target_data, orient="index", columns=columns)
        target_fold.index.name = original_fold.index.name
        return target_fold

    def assign_missing_patient_ids(self) -> None:
        original_folds = self.data.folds["scan"]

        self.missing_patient_ids_map = {}
        for fold_version in original_folds:
            for mode in original_folds[fold_version]:
                self.assign_missing_patient_ids_for_mode(fold_version, mode)

    def assign_missing_patient_ids_for_mode(self, fold_version: str, mode: str) -> None:
        # region Find existing patient ids first
        original_folds = self.data.folds["scan"][fold_version][mode]

        existing_patients: list[str] = []
        for original_fold_path in original_folds:
            original_fold = pd.read_csv(original_fold_path, index_col=SCAN_ID)
            for row_id in original_fold.index:
                patient_id, *_ = row_id.split("_")

                if patient_id not in existing_patients:
                    existing_patients.append(patient_id)
        # endregion

        # region Assign (round-robin)
        current_partition = 0
        partition_count = len(original_folds)
        missing_patient_ids_map: dict[str, int] = {}
        for bounding_boxes_id in self.data.extracted_breasts_images:
            patient_id, *_ = bounding_boxes_id.split("_")
            if patient_id in existing_patients:
                continue

            missing_patient_ids_map[patient_id] = current_partition
            current_partition = (current_partition + 1) % partition_count
        # endregion

        # region Save in self.missing_patient_ids_map
        if fold_version not in self.missing_patient_ids_map:
            self.missing_patient_ids_map[fold_version] = {mode: missing_patient_ids_map}
        else:
            self.missing_patient_ids_map[fold_version][mode] = missing_patient_ids_map
        # endregion

    def convert_fold_for_breasts_bounding_boxes(self,
                                                original_fold: pd.DataFrame,
                                                fold_version: str,
                                                mode: str,
                                                fold_number: int,
                                                ) -> pd.DataFrame:
        # region Get mapping for existing patients
        patients_subsets: dict[str, str] = {}
        for row_id, row in original_fold.iterrows():
            row_id: str
            patient_id, *_ = row_id.split("_")
            subset_id = row[SUBSET_ID]
            if patient_id not in patients_subsets:
                patients_subsets[patient_id] = subset_id

            elif patients_subsets[patient_id] != subset_id:
                raise RuntimeError(subset_id, patients_subsets[patient_id])
        # endregion

        target_data = {}
        original_folds = self.data.folds["scan"][fold_version][mode]
        partition_count = len(original_folds)
        for bounding_boxes_id in self.data.extracted_breasts_images:
            # region Get images/masks paths
            images_paths = self.data.extracted_breasts_images_paths[bounding_boxes_id]
            if bounding_boxes_id in self.data.extracted_breasts_masks_paths:
                masks_paths = self.data.extracted_breasts_masks_paths[bounding_boxes_id]
            else:
                masks_paths = ""
            # endregion

            # region Get subset ID for patient
            patient_id, *_ = bounding_boxes_id.split("_")
            if patient_id in patients_subsets:
                subset_id = patients_subsets[patient_id]

            elif patient_id in self.missing_patient_ids_map[fold_version][mode]:
                # noinspection PyUnresolvedReferences
                partition_id = self.missing_patient_ids_map[fold_version][mode][patient_id]
                if partition_id == fold_number:
                    subset_id = "validation"
                elif partition_id == ((fold_number + 1) % partition_count):
                    subset_id = "test"
                else:
                    subset_id = "train"

            else:
                raise RuntimeError(patient_id)
            # endregion

            row = {
                SUBSET_ID: subset_id,
                DEFAULT_IMAGE_COLUMN: images_paths[-1],
                DEFAULT_MASK_COLUMN: masks_paths[0],
            }
            target_data[bounding_boxes_id] = row

        target_fold = pd.DataFrame.from_dict(data=target_data, orient="index")
        target_fold.index.name = original_fold.index.name
        return target_fold

    # region Folds paths
    def get_scan_folds_path(self) -> Path:
        return self.get_folds_output_path("scan")

    def register_folds(self,
                       modality: str,
                       fold_version: str,
                       mode: str,
                       folds_paths: list[Path]):
        if modality not in self.data.folds:
            self.data.folds[modality] = {}

        if fold_version not in self.data.folds[modality]:
            self.data.folds[modality][fold_version] = {}

        self.data.folds[modality][fold_version][mode] = folds_paths
    # endregion
