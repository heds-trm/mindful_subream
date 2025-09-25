import numpy as np
import torch
from monai.transforms import LoadImage, ScaleIntensity, Resize, Compose
import cv2
from pathlib import Path
import argparse

from mindful_core.data.transforms.imaging import LoadImage4D


def to_image_2d(sample: torch.Tensor) -> np.ndarray:
    return sample[-1, ..., 35].numpy()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--reference_path", required=True, type=str)
    arg_parser.add_argument("--phases_path", required=True, type=str)
    args = arg_parser.parse_args()

    reference_path = Path(args.reference_path)
    phases_path = Path(args.phases_path)

    transform_3d = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        Resize(spatial_size=(512, 512, 70))
    ])

    transform_4d = Compose([
        LoadImage4D(image_only=True),
        ScaleIntensity(channel_wise=True),
        Resize(spatial_size=(512, 512, 70))
    ])

    for sample_3d_path in reference_path.iterdir():
        sample_4d_path = phases_path / sample_3d_path.stem

        sample_3d = transform_3d(sample_3d_path)
        sample_4d = transform_4d(sample_4d_path)

        sample_3d = to_image_2d(sample_3d)
        sample_4d = to_image_2d(sample_4d)

        cv2.imshow(sample_3d_path.stem + "_3d", sample_3d)
        cv2.imshow(sample_3d_path.stem + "_4d", sample_4d)
        cv2.imshow(sample_3d_path.stem + "_delta", np.abs(sample_4d - sample_3d))
        cv2.waitKey()


if __name__ == "__main__":
    main()
