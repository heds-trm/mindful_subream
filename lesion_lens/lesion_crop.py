import SimpleITK
import torch
import numpy as np
from pathlib import Path

from mindful_core.utils.dicom import load_dicom_sitk

from lesion_lens.warnings_context_manager import WarningsContextManager


class LesionCrop(object):
    def __init__(self, image: SimpleITK.Image, lesion_position: np.ndarray, crop_size: float = 50.0) -> None:
        self.lesion_position = lesion_position
        image_size: tuple[int, ...] = image.GetSize()

        if len(image_size) == 4:
            if image_size[-1] != 1:
                WarningsContextManager.warn("Your image appears to have 4 dimensions, with size being {}. "
                                            "Removing last dimension by taking the last channel.".format(image_size))
            image = image[..., -1]

        elif len(image_size) != 3:
            raise RuntimeError("Your image have {} dimensions but expected a 3D image.".format(len(image_size)))

        # noinspection PyTypeChecker
        self.original_image_size: tuple[int, int, int] = image_size

        start = image.TransformPhysicalPointToIndex([float(x) - crop_size / 2.0 for x in lesion_position])
        end = image.TransformPhysicalPointToIndex([float(x) + crop_size / 2.0 for x in lesion_position])

        self.crop_spatial_shape = tuple([_end - _start for _start, _end in zip(start, end)])

        upper_padding, lower_padding = None, None
        if any([index < 0 for index in start]):
            upper_padding = [max(0, -index) for index in start]
            start = [max(0, index) for index in start]

        if any([index >= dim for index, dim in zip(end, self.original_image_size)]):
            lower_padding = [max(0, index - dim) for index, dim in zip(end, self.original_image_size)]
            end = [min(dim, index) for index, dim in zip(end, self.original_image_size)]

        self.start: tuple[int, int, int] = start
        self.end: tuple[int, int, int] = end
        self.crop_slices = [slice(_start, _end) for _start, _end in zip(self.start, self.end)]

        image = image[self.crop_slices]

        self.upper_padding = upper_padding
        self.lower_padding = lower_padding
        if self.pad_lesion:
            upper_padding = upper_padding or [0, 0, 0]
            lower_padding = lower_padding or [0, 0, 0]
            image = SimpleITK.ConstantPad(image, upper_padding, lower_padding)

        self.image = image

    @staticmethod
    def _check_lesion_position(image: SimpleITK.Image, lesion_position: np.ndarray, lesion_id: str) -> None:
        point_index = image.TransformPhysicalPointToIndex([float(x) for x in lesion_position])
        image_size: tuple[int, int, int] = image.GetSize()

        below_zero = [x < 0 for x in point_index]
        above_size = [x >= dim for x, dim in zip(point_index, image_size)]

        if not (any(below_zero) or any(above_size)):
            # No problem, return
            return
        
        states = []
        names = ["x", "y", "z"]
        for i, (x, x_below, x_above) in enumerate(zip(lesion_position, below_zero, above_size)):
            if x_below:
                if x_above:
                    state = "Empty dimension"
                else:
                    state = "Below minimum"
            else:
                if x_above:
                    state = "Above maximum"
                else:
                    state = "Within bounds"
            states.append("Dim {} ({}={}mm): `{}`".format(i, names[i], x, state))
        states_message = ", ".join(states)

        outside = [x_below or x_above for x_below, x_above in zip(below_zero, above_size)]
        count = "All" if all(outside) else sum(outside)
        message = ("{} lesion coordinates were found to be oustide of the image for lesion `{}` ({}). "
                   "This may result in incorrect results.".
                   format(count, lesion_id, states_message))
            
        WarningsContextManager.warn(message)

    @property
    def pad_lesion(self) -> bool:
        return (self.upper_padding is not None) or (self.lower_padding is not None)

    def remove_padding(self, image: torch.Tensor | SimpleITK.Image) -> torch.Tensor | SimpleITK.Image:
        start = self.upper_padding or [None, None, None]
        if self.lower_padding is None:
            end = [None, None, None]
        else:
            end = [-x if x > 0 else None for x in self.lower_padding]
        slices = [slice(_start, _end) for _start, _end in zip(start, end)]
        return image[slices]


def crop_lesion(image: Path | str | SimpleITK.Image,
                lesion_position: np.ndarray,
                lesion_id: str,
                verbose: bool = False
                ) -> tuple[Path, LesionCrop]:
    if not isinstance(image, SimpleITK.Image):
        image = load_dicom_sitk(image)

    LesionCrop._check_lesion_position(image, lesion_position, lesion_id)

    image_size: tuple[int, ...] = image.GetSize()
    if len(image_size) == 4:
        if image_size[-1] != 1:
            WarningsContextManager.warn("Your image appears to have 4D, with size being {}. "
                                        "Removing last dimension by taking the last channel.".format(image_size))
        image = image[..., -1]

    lesion_crop = LesionCrop(image, lesion_position, crop_size=50.0)

    output_path = Path("/tmp", "{}.mha".format(lesion_id))
    if verbose:
        print("Writing temporary lesion crop for lesion `{}` to {}".format(lesion_id, output_path.absolute()))

    SimpleITK.WriteImage(lesion_crop.image, output_path)
    return output_path, lesion_crop
