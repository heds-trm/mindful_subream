import numpy as np
import os
import argparse

from mindful_core.utils.parsing import safe_bool, safe_int

from mindful_subream.dataset.build.build_steps.folds_builder import (
    load_labels,
    get_patients,
    balance_lymphnodes,
    update_labels,
    stochastic_folds_generation,
    patient_to_samples_folds,
    make_cross_validation_folds,
    save_cross_validation_folds
)


def main():
    # region Arg parse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--roots", required=True, nargs="+")
    arg_parser.add_argument("--output_folder", default=None)

    arg_parser.add_argument("--labels_file", required=True, type=str)
    arg_parser.add_argument("--labels_mode", choices=["b_g_m", "bg_m", "b_m", "bm_g", "b_g"], required=True)

    arg_parser.add_argument("--folds_count", default=5, type=int)
    arg_parser.add_argument("--test_folds_count", default=1, type=int)
    arg_parser.add_argument("--validation_folds_count", default=1, type=int)
    arg_parser.add_argument("--iterations", default=100000, type=int)
    arg_parser.add_argument("--balance_lymphnodes", default=False, type=bool)
    arg_parser.add_argument("--seed", default=3930787959)
    args = arg_parser.parse_args()
    roots: list[str] = args.roots
    output_folder: str = args.output_folder
    labels_file: str = args.labels_file
    labels_mode: str = args.labels_mode
    folds_count: int = safe_int(args.folds_count, default=5)
    test_folds_count: int = safe_int(args.test_folds_count, default=1)
    validation_folds_count: int = safe_int(args.validation_folds_count, default=1)
    iterations: int = safe_int(args.iterations, default=100000)
    do_balance_lymphnodes: bool = safe_bool(args.balance_lymphnodes, default=False)
    seed: int = safe_int(args.seed, default=3930787959)
    # endregion
    np.random.seed(seed)

    labels = load_labels(labels_file, labels_mode)
    patients = sum([get_patients(root) for root in roots], [])
    if do_balance_lymphnodes:
        patients = balance_lymphnodes(patients)
    patients = update_labels(patients, labels)

    patient_folds = stochastic_folds_generation(patients, folds_count, iterations)
    sample_folds = patient_to_samples_folds(patient_folds)

    if output_folder is None:
        output_folder = os.path.commonpath(roots)
    output_folder = os.path.join(output_folder, labels_mode)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("Writing folds to", output_folder)
    data_frames = make_cross_validation_folds(sample_folds, test_folds_count, validation_folds_count)
    save_cross_validation_folds(data_frames, output_folder)


if __name__ == "__main__":
    main()
