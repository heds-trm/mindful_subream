import torch
from monai.transforms import EnsureChannelFirstd, Compose, RandAffined, RandRotated, RandFlipd, EnsureTyped
import SimpleITK
import numpy as np
from multiprocessing.pool import Pool
from tqdm import tqdm
import os
import time

from pathlib import Path

from mindful_subream.dataset.build.build_steps import SubreamBuildStep, SubreamROIExtractor
from mindful_subream.dataset.build.build_data import SubreamBuildData
from mindful_subream.dataset.build.build_logger import SubreamBuildLogger
from mindful_subream.dataset.roi_extraction import save_rois


class SubreamMIPExtractor(SubreamBuildStep):
    def __init__(self,
                 build_data: SubreamBuildData,
                 output_root: Path,
                 logger: SubreamBuildLogger):
        super(SubreamMIPExtractor, self).__init__(build_data, output_root, logger)
        self._keys: list[str] | None = None
        self._augmentations: Compose | None = None

    def run(self) -> None:
        multithreaded = False
        if multithreaded:
            self.extract_mips_multithreaded()
        else:
            self.extract_mips()
        self.log("SUCCESS: MIPs successfully processed.")

        self.save_mips()
        self.log("SUCCESS: MIPs successfully saved to disk.")

    def extract_mips(self):
        keys, augmentations = self.get_augmentations(phase_count=13)
        shared_parameters = {
            "keys": keys,
            "augmentations": augmentations,
            "augmentation_count": self.data.mip_augmentation_count,
            "roi_size": self.data.roi_size,
            "mip_dim": self.data.mip_dim
        }
        patients_points = SubreamROIExtractor.as_patient_lesion_dict(self.data.masks_roi_points)
        ids_progress_bar = tqdm(self.data.patients_ids)

        extracted_mips_per_patient: list[dict[str, SimpleITK.Image]] = []
        for patient_id in ids_progress_bar:
            if not self.data.has_patient_images(patient_id):
                continue

            ids_progress_bar.set_description("Extracting MIPs for patient `{}`".format(patient_id))
            patient_mips = self._extract_patient_mips(patient_id=patient_id,
                                                      images=self.data.read_patient_images(patient_id),
                                                      base_patient_rois=self.get_patient_rois(patient_id),
                                                      patient_masks_roi_points=patients_points[patient_id],
                                                      **shared_parameters)
            extracted_mips_per_patient.append(patient_mips)

        extracted_mips: dict[str, SimpleITK.Image] = {mip_id: mip
                                                      for extracted_patient_mips in extracted_mips_per_patient
                                                      for mip_id, mip in extracted_patient_mips.items()}
        self.data.extracted_mips = extracted_mips

    def extract_mips_multithreaded(self):
        processes_count = min(os.cpu_count(), len(self.data.patients_ids))

        keys, augmentations = self.get_augmentations(phase_count=13)
        shared_parameters = {
            "keys": keys,
            "augmentations": augmentations,
            "augmentation_count": self.data.mip_augmentation_count,
            "roi_size": self.data.roi_size,
            "mip_dim": self.data.mip_dim
        }

        patients_points = SubreamROIExtractor.as_patient_lesion_dict(self.data.masks_roi_points)
        with Pool(processes=processes_count) as pool:
            loading_description = "Loading images and starting {} threads to process faster".format(processes_count)
            load_progress_bar = tqdm(self.data.patients_ids, desc=loading_description)
            async_results = [pool.apply_async(func=self._extract_patient_mips, kwds={
                "patient_id": patient_id,
                "images": self.data.read_patient_images(patient_id),
                "base_patient_rois": self.get_patient_rois(patient_id),
                "patient_masks_roi_points": patients_points[patient_id],
                **shared_parameters
            })
                             for patient_id in load_progress_bar
                             if self.data.has_patient_images(patient_id)
                             ]

            previous_ready_count = 0
            k = 0
            print("MIP generation can take a while before any progress is shown.")
            mip_progress_bar = tqdm(self.data.patients_ids, desc="Generating MIPs...")
            while not all([async_result.ready() for async_result in async_results]):
                ready_count = sum([async_result.ready() for async_result in async_results])
                mip_progress_bar.update(ready_count - previous_ready_count)
                previous_ready_count = ready_count
                time.sleep(1.0)
                mip_progress_bar.set_description("Generating MIPs" + '.' * (k + 1) + ' ' * (2 - k))
                k = (k + 1) % 3

        extracted_mips_per_patient = [async_result.get() for async_result in async_results]

        # extracted_mips_per_patient = [self.extract_patient_mips(patient_id) for patient_id in progress_bar]
        extracted_mips: dict[str, SimpleITK.Image] = {mip_id: mip
                                                      for extracted_patient_mips in extracted_mips_per_patient
                                                      for mip_id, mip in extracted_patient_mips.items()}
        self.data.extracted_mips = extracted_mips

    @staticmethod
    def _extract_patient_mips(patient_id: str,
                              images: list[SimpleITK.Image],
                              base_patient_rois: dict[str, list[SimpleITK.Image]],
                              keys: list[str],
                              augmentations: Compose,
                              augmentation_count: int,
                              patient_masks_roi_points: dict[str, np.ndarray],
                              roi_size: float,
                              mip_dim: int,
                              ) -> dict[str, SimpleITK.Image]:
        extracted_mips: dict[str, SimpleITK.Image] = {}
        for i in range(augmentation_count + 1):
            if i == 0:
                rois = base_patient_rois
            else:
                try:
                    augmented_images = SubreamMIPExtractor._augment_images(images, keys, augmentations)
                except RuntimeError as error:
                    raise RuntimeError("Error with patient_id `{}`: {}".format(patient_id, error))

                rois = SubreamROIExtractor.extract_patient_rois(patient_id,
                                                                augmented_images,
                                                                patient_masks_roi_points,
                                                                roi_size)

            mips = extract_mip_stacks(rois, dim=mip_dim)
            for roi_id, mip in mips.items():
                augmented_roi_id = "{}_{}".format(roi_id, i)
                extracted_mips[augmented_roi_id] = mip

        return extracted_mips

    def get_patient_rois(self, patient_id: str) -> dict[str, list[SimpleITK.Image]]:
        return {roi_id: roi for roi_id, roi in self.data.extracted_rois.items()
                if roi_id.startswith(patient_id + "_")}

    # region Augmentations
    # def augment_images(self, images: list[SimpleITK.Image]) -> list[SimpleITK.Image]:
    #     keys, augmentations = self.get_augmentations(len(images))
    #     return self._augment_images(images, keys, augmentations)

    @staticmethod
    def _augment_images(images: list[SimpleITK.Image],
                        keys: list[str],
                        augmentations: Compose,
                        ) -> list[SimpleITK.Image]:
        inputs = {key: SubreamMIPExtractor.tensor_from_image(image) for key, image in zip(keys, images)}
        images_dict: dict[str, torch.Tensor] = augmentations(inputs)
        images = [torch_image_to_simpleitk(image) for image in images_dict.values()]
        return images

    @staticmethod
    def tensor_from_image(image: SimpleITK.Image, to_gpu: bool = False) -> torch.Tensor:
        tensor = torch.as_tensor(SimpleITK.GetArrayFromImage(image))
        # tensor = itk_image_to_metatensor(image)
        if to_gpu:
            tensor = tensor.cuda()
        return tensor

    def get_augmentations(self, phase_count: int) -> tuple[list[str], Compose]:
        if self._augmentations is None:
            self.create_default_transforms(phase_count)
        else:
            self.check_phase_count_with_augmentations(phase_count)
        return self._keys, self._augmentations

    def check_phase_count_with_augmentations(self, phase_count: int) -> None:
        if len(self._keys) != phase_count:
            self.error(
                message="Number of images ({}) different from expected ({})".format(phase_count, len(self._keys)))

    def create_default_transforms(self,
                                  phase_count: int,
                                  max_rotation: float = 0.34,
                                  ) -> None:
        self._keys, self._augmentations = self._create_default_transforms(phase_count, max_rotation)

    @staticmethod
    def _create_default_transforms(phase_count: int,
                                   max_rotation: float = 0.34,
                                   ) -> tuple[list[str], Compose]:
        keys = [f"img_{i}" for i in range(phase_count)]
        augmentations = Compose(
            [
                EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
                RandRotated(keys=keys, range_x=max_rotation, range_y=max_rotation, range_z=max_rotation,
                            prob=0.8, dtype=None),
                RandAffined(keys=keys, prob=0.5, translate_range=(2, 2, 2), padding_mode="border"),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # Left-Right flip
                EnsureTyped(keys=keys),
            ]
        )
        return keys, augmentations

    # endregion

    # region Saving
    def save_mips(self):
        extracted_mip_paths: dict[str, Path] = save_rois(output_images_folder=self.get_images_output_path("mip"),
                                                         rois=self.data.extracted_mips,
                                                         short_labels=self.data.get_lesion_labels_by_id(),
                                                         use_progress_bar=True)
        self.data.extracted_mip_paths = extracted_mip_paths

    # endregion


# region MIP extraction
def extract_mip_stacks(samples: dict[str, list[SimpleITK.Image]],
                       dim: int
                       ) -> dict[str, SimpleITK.Image]:
    return {sample_id: extract_mip_stack(phases, dim)
            for sample_id, phases in samples.items()}


def extract_mip_stack(phases: list[SimpleITK.Image], dim: int) -> SimpleITK.Image:
    phase_count = len(phases)
    reference = phases[0]
    output_shape = (reference.GetWidth(), reference.GetHeight(), phase_count)
    mip_phase_shape = (reference.GetWidth(), reference.GetHeight(), 1)

    mip_stack = SimpleITK.Image(output_shape, reference.GetPixelID())
    for i, phase in enumerate(phases):
        mip_phase = extract_mip_phase(phase, dim)
        mip_stack = SimpleITK.Paste(mip_stack, mip_phase, mip_phase_shape, [0] * 3, [0, 0, i])
    mip_stack.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    return mip_stack


def extract_mip_phase(image: SimpleITK.Image,
                      dim: int) -> SimpleITK.Image:
    direction = image.GetDirection()
    dim_size = image.GetSize()[dim]

    if (dim_size % 2) == 1:
        origin_index = [0, 0, (dim_size - 1) // 2]
        origin = image.TransformIndexToPhysicalPoint(origin_index)
    else:
        origin_index_0 = [0, 0, dim_size // 2 - 1]
        origin_index_1 = [0, 0, dim_size // 2]
        origin_0 = image.TransformIndexToPhysicalPoint(origin_index_0)
        origin_1 = image.TransformIndexToPhysicalPoint(origin_index_1)
        origin = np.mean([origin_0, origin_1], axis=0)

    image = SimpleITK.MaximumProjection(image, projectionDimension=dim)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


# endregion

def torch_image_to_simpleitk(image: torch.Tensor, original: SimpleITK.Image = None) -> SimpleITK.Image:
    image = torch.squeeze(image, dim=0)
    image = image.cpu().numpy()
    output = SimpleITK.GetImageFromArray(image)
    if original is not None:
        output.SetSpacing(original.GetSpacing())
        output.SetOrigin(original.GetOrigin())
        output.SetDirection(original.GetDirection())
    return output
