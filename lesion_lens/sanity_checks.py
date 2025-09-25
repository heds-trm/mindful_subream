from pathlib import Path


def check_for_output_data(outputs_folder: Path, image_uuid: str):
    """
    Checks outputs present in the output folder to ensure all required data have been exported.

    :param outputs_folder:
    :param image_uuid:
    :return:
    """
    required_outputs = [
        outputs_folder / "report.pdf",
        outputs_folder / "classified-lesions-breast.json",
        outputs_folder / "images" / "occlusion_sensitivity" / "{}.mha".format(image_uuid)
    ]

    missing_outputs = [output for output in required_outputs if not output.exists()]
    if len(missing_outputs) > 0:
        raise RuntimeError("Missing the following outputs: {}".format(missing_outputs))
