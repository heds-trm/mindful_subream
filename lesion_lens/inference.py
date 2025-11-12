from mindful_core.utils.redirect_io import stderr_redirected, stdout_redirected

with stdout_redirected(), stderr_redirected():
    import SimpleITK
    import numpy as np
    import torch
    import os
    from pathlib import Path
    import warnings
    import argparse

    from mindful_core.utils.dicom import DICOMAttributesCollection, load_dicom_sitk
    from mindful_core.utils.misc import find_first_path, try_load_json
    from mindful_core.experiments.inference import restore_model, load_sample
    from mindful_core.data.modalities import ModalityType
    from mindful_core.models.model_output import ClassifierOutput
    from mindful_core.analysis.visualization import VisualizerGroup, Visualizer

    from mindful_subream.lesion_lens.warnings_context_manager import WarningsContextManager
    from mindful_subream.lesion_lens.parsing import (parse_suspected_lesions_positions,
                                                     parse_scalar_data,
                                                     parse_categorical_data)
    from mindful_subream.lesion_lens.lesion_crop import LesionCrop, crop_lesion
    from mindful_subream.lesion_lens.reporting import NipplesPosition, LesionPointPrediction, export_point_predictions
    from mindful_subream.lesion_lens.reporting import export_occlusion_map
    from mindful_subream.lesion_lens.reporting import LesionLensPDFReport
    from mindful_subream.lesion_lens.sanity_checks import check_for_output_data


class InferenceException(Exception):
    """
        Used to differentiate exceptions created by this script for users and other exceptions.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return "LesionLens encountered the following error during inference: {}".format(self.message)


# region Run
def batch_sample(sample: tuple[torch.Tensor, ...] | torch.Tensor
                 ) -> tuple[torch.Tensor, ...] | torch.Tensor:
    if isinstance(sample, torch.Tensor):
        sample = sample.unsqueeze(dim=0)
    else:
        sample = tuple([modality.unsqueeze(dim=0) for modality in sample])
    return sample


def run_for_lesion(inputs: dict[str, str | Path],
                   metadata_path: Path,
                   checkpoint_path: Path,
                   output_folder: str | Path,
                   visualizers_path: str | Path,
                   lesion_id: str
                   ) -> tuple[dict[str, np.ndarray], dict[Visualizer, dict[str, torch.Tensor]]]:
    print("-- Running inference for lesion `{}`...".format(lesion_id))

    use_gpu = torch.cuda.device_count() > 0
    output_folder = Path(output_folder)
    output_image_folder = output_folder / "images"
    output_image_folder.mkdir(exist_ok=True)

    model, _, preproc_pipeline = restore_model(metadata_path, checkpoint_path)
    model_input_modalities = preproc_pipeline.output_modalities - ModalityType.LABEL

    model.eval()
    if use_gpu:
        model.cuda()

    sample = load_sample(inputs, preproc_pipeline, to_gpu=use_gpu)
    # meta_data: tuple[dict[str, Any], ...] = tuple(tensor.meta for tensor in sample)
    batched_sample = batch_sample(sample)

    with torch.no_grad():
        prediction: ClassifierOutput = model(batched_sample)
        if prediction.batched:
            prediction = prediction[0]

    # region Visualization (interpretability)
    visualizers = VisualizerGroup(visualizers_path,
                                  model=model,
                                  log_dir=output_folder,
                                  input_modalities=model_input_modalities,
                                  )
    visualizations = visualizers(sample, prediction=prediction)
    # endregion

    # region Predictions
    logits = prediction.logits.detach().cpu().numpy()
    probabilities = prediction.get_probabilities().detach().cpu().numpy()

    formatted_logits = [str(round(x, ndigits=3)) for x in logits]
    formatted_probabilities = ["{}%".format(round(x * 100, ndigits=1)) for x in probabilities]
    print("Logits: {}".format(", ".join(formatted_logits)))
    print("Probabilities: {}".format(", ".join(formatted_probabilities)))

    print("-" * 10)

    values: dict[str, np.ndarray] = {
        "logits": logits,
        "probabilities": probabilities,
    }
    # endregion

    return values, visualizations


# endregion


def find_input_image(images_root: Path) -> tuple[Path, bool]:
    image_path = find_first_path(images_root, pattern="*.mha")
    if image_path is None:
        image_path = find_first_path(images_root, pattern="*.dcm", recursive=True)
        using_dicom = True
    else:
        using_dicom = False

    if image_path is None:
        error = "Could not find an image in `{}`. ".format(images_root)
        if images_root.exists():
            other_items = list(images_root.rglob(pattern="*"))
            error += "Found {} other items: {}.".format(len(other_items), other_items)
        else:
            error += "Path does not exist."
        raise InferenceException(error)

    if using_dicom:
        dicom_in_dir = list(image_path.parent.glob(pattern="*.dcm"))
        if len(dicom_in_dir) > 0:
            image_path = image_path.parent

    return image_path, using_dicom


# region Checks
# region Data checks
def check_dicom_metadata(dicom_data: DICOMAttributesCollection):
    if not dicom_data.has_dicom_attribute("Pixel Spacing"):
        WarningsContextManager.warn("No `Pixel Spacing` attribute found in the DICOM file. "
                                    "This will default to (1.0, 1.0, 1.0) and may lead to incorrect results.")

    has_position = (dicom_data.has_dicom_attribute("Patient Position") or
                    dicom_data.has_dicom_attribute("Image Position (Patient)"))
    if not has_position:
        WarningsContextManager.warn("Missing both `Patient Position` and `Image Position (Patient)`"
                                    " attributes in the DICOM file."
                                    " This will default to an origin of (0.0, 0.0, 0.0)"
                                    " and may lead to incorrect results.")


def check_image(image: SimpleITK.Image) -> SimpleITK.Image:
    image_size: tuple[int, ...] = image.GetSize()

    if len(image_size) == 4:
        if image_size[-1] != 1:
            WarningsContextManager.warn("Your image appears to have 4 dimensions, with size being {}. "
                                        "Removing last dimension by taking the last channel.".format(image_size))
        image = image[..., -1]

    elif len(image_size) != 3:
        raise RuntimeError("Your image have {} dimensions but expected a 3D image.".format(len(image_size)))

    return image


# endregion
# endregion


def run_inference(inputs_folder: str | Path = "/input/",
                  outputs_folder: str | Path = "/output/",
                  model_folder: str | Path = "/opt/ml/model/",
                  ) -> None:
    inputs_folder = Path(inputs_folder)
    outputs_folder = Path(outputs_folder)
    model_folder = Path(model_folder)

    # region Inputs
    images_root = inputs_folder / "images/ultrafast-subtraced-mri"
    image_path, using_dicom = find_input_image(images_root)
    if using_dicom:
        dicom_data = DICOMAttributesCollection(image_path)
        check_dicom_metadata(dicom_data)
    else:
        dicom_data = None

    image_uuid = image_path.stem
    suspected_lesions_positions_path = inputs_folder / "suspected-lesions-breast.json"
    patient_information_path = inputs_folder / "patient-information-subream-algorithm.json"
    lesion_information_path = inputs_folder / "lesion-information-subream-algorithm.json"

    suspected_lesions_positions = parse_suspected_lesions_positions(suspected_lesions_positions_path)
    suspected_lesions_positions, nipple_positions = NipplesPosition.from_lesion_list(suspected_lesions_positions)
    categorical_data = parse_categorical_data(patient_information_path)
    scalar_data = parse_scalar_data(patient_information_path, lesion_information_path, suspected_lesions_positions)
    # endregion

    # region Outputs
    predictions_path = outputs_folder / "classified-lesions-breast.json"
    occlusion_map_dir = outputs_folder / "images" / "occlusion_sensitivity"
    occlusion_map_filename = occlusion_map_dir / image_uuid
    pdf_path = outputs_folder / "report.pdf"

    lesions_point_predictions: dict[str, LesionPointPrediction] = {}
    classes = ["benign", "lymph node", "malignant"]
    occlusion_maps: dict[str, torch.Tensor] = {}
    # endregion

    # region Inference/model config
    use_unimodal_model = (categorical_data is None) or (scalar_data is None)
    if use_unimodal_model:
        model_folder = model_folder / "unimodal"
    else:
        model_folder = model_folder / "multimodal"

    metadata_path = find_first_path(model_folder, "*.pkl")
    checkpoint_path = find_first_path(model_folder, "*.ckpt")
    visualizers_path = find_first_path(model_folder.parent, "*.json")
    # endregion

    image = load_dicom_sitk(image_path)
    image = check_image(image)

    # region Main inference loop
    lesion_crops: dict[str, LesionCrop] = {}
    for lesion_id in suspected_lesions_positions:
        # region 1 - Get lesion inputs
        lesion_position = suspected_lesions_positions[lesion_id]
        lesion_crop_path, lesion_crop = crop_lesion(image, lesion_position, lesion_id)
        lesion_crops[lesion_id] = lesion_crop
        inputs = {
            "image": lesion_crop_path,
        }
        if not use_unimodal_model:
            optional_inputs = {
                "scalar": scalar_data[lesion_id],
                "categorical": categorical_data,
            }
            inputs = {**inputs, **optional_inputs}
        # endregion
        # region 2 - Run inference
        lesion_outputs = run_for_lesion(inputs, metadata_path, checkpoint_path, outputs_folder, visualizers_path,
                                        lesion_id)
        # endregion
        # region 3 - Aggregate outputs
        lesion_predictions, lesion_visualizations = lesion_outputs

        # region Format predictions
        probabilities = lesion_predictions["probabilities"]
        max_class = probabilities.argmax()
        lesion_class_name = classes[max_class]
        lesion_point = list([float(x) for x in lesion_position])
        lesion_probability = float(probabilities[max_class])
        # endregion

        lesion_point_prediction = LesionPointPrediction(lesion_class_name, lesion_point, lesion_probability)
        lesions_point_predictions[lesion_id] = lesion_point_prediction

        occlusions = list(lesion_visualizations.values())[0]
        occlusion_maps[lesion_id] = occlusions["image"]

        # endregion
        # region 4 - Clean up
        os.remove(lesion_crop_path)
        # endregion
    # endregion

    # region Finally, export results
    export_point_predictions(predictions_path, list(lesions_point_predictions.values()))
    export_occlusion_map(occlusion_maps, lesion_crops, image_path, occlusion_map_filename)

    patient_data = try_load_json(patient_information_path, "Patient Information")
    lesions_geometry_list = try_load_json(lesion_information_path, "Lesion Geometric Data")["lesions"]
    lesions_geometry = {}
    for lesion_geometry in lesions_geometry_list:
        lesion_id = lesion_geometry.pop("name")
        lesions_geometry[lesion_id] = lesion_geometry["geometry"]

    if dicom_data is not None:
        dicom_data = DICOMAttributesCollection(image_path)
        patient_data["name"] = dicom_data.patient_name
        patient_data["birthdate"] = dicom_data.patient_birth_date
        patient_data["acquisition_date"] = dicom_data.acquisition_date

    report = LesionLensPDFReport(patient_data=patient_data,
                                 lesion_crops=lesion_crops,
                                 lesions_geometry=lesions_geometry,
                                 predictions=lesions_point_predictions,
                                 occlusion_maps=occlusion_maps,
                                 nipples_position=nipple_positions,
                                 templates_root=Path("./subream/lesion_lens").absolute())
    report.export(pdf_path)
    # endregion

    check_for_output_data(outputs_folder, image_uuid)

    print("Successfully completed inference!")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", type=str, default="/input/")
    arg_parser.add_argument("--output", type=str, default="/output/")
    arg_parser.add_argument("--model", type=str, default="/opt/ml/model/")

    args = arg_parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    model_folder = Path(args.model)

    try:
        with WarningsContextManager() as warnings_context_manager, stderr_redirected():
            run_inference(input_folder, output_folder, model_folder)
    except InferenceException as exception:
        error_message = str(exception)
    else:
        error_message = None

    if warnings_context_manager.has_warnings:
        warning_messages = ["\t{}".format(message) for message in warnings_context_manager.warnings]
        warning_messages.insert(0, "LesionLens encountered the following user warnings:")
        warnings.warn("\n".join(warning_messages))

    if error_message is not None:
        exit_code = 1
    else:
        exit_code = 0

    exit(exit_code)


if __name__ == "__main__":
    main()
