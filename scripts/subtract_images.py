import argparse
import SimpleITK as sitk
from pathlib import Path


def check_same_geometry(im1, im2):
    """
    Ensure two images have identical geometry.
    """
    if im1.GetSize() != im2.GetSize():
        raise ValueError("Images have different sizes.")

    if im1.GetSpacing() != im2.GetSpacing():
        raise ValueError("Images have different spacing.")

    if im1.GetOrigin() != im2.GetOrigin():
        raise ValueError("Images have different origins.")

    if im1.GetDirection() != im2.GetDirection():
        raise ValueError("Images have different directions.")


def subtract_images(im1_path, im2_path, output_path):
    """
    Compute im1 - im2 and save as float compressed MHA.
    """
    im1_path = Path(im1_path)
    im2_path = Path(im2_path)
    output_path = Path(output_path)

    # Read images
    im1 = sitk.ReadImage(str(im1_path))
    im2 = sitk.ReadImage(str(im2_path))

    # Check geometry
    check_same_geometry(im1, im2)

    # Cast to float
    im1 = sitk.Cast(im1, sitk.sitkFloat32)
    im2 = sitk.Cast(im2, sitk.sitkFloat32)

    # Subtract
    result = sitk.Subtract(im1, im2)

    # Ensure output folder exists
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save compressed
    sitk.WriteImage(result, str(output_path), useCompression=True)

    print(f"Saved result: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Subtract two MHA images (im1 - im2) with float conversion."
    )

    parser.add_argument("--im1", required=True, help="First input image (im1)")
    parser.add_argument("--im2", required=True, help="Second input image (im2)")
    parser.add_argument("--output", required=True, help="Output MHA file")

    args = parser.parse_args()

    subtract_images(args.im1, args.im2, args.output)


if __name__ == "__main__":
    main()