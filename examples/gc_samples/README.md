# Lesion Lens

![LesionLensLogo](../doc/Logo_LesionLens_small.png)

The Lension Lens algorithm is hosted on the [Grand-Challenge](https://grand-challenge.org/algorithms/subream-breast-mri-lesion-lens/) platform.

The Lesion Lens algorithm aims at classifying suspected breast lesions into malignant, benign or lymph nodes based on a subtracted image acquired with an ultrafast dynamic contrast-enhanced (UF-DCE) MRI sequence. Since suspected lesions are not automatically detected, their locations must be provided as an input to the algorithm.

Lesion Lens is a _multimodal_ model that considers three types of data modalities: 
- **imaging**: a subtracted image based on the last and first phases of the UF-DCE sequence.
- **categorical data**: information relative to the patient known to be associated with cancer risk.
- **scalar data**: geometrical information derived from segmented lesions.

This data must be passed to the algorithm via a user interface provided by Grand-Challenge as explained in the [algorithm page](https://grand-challenge.org/algorithms/subream-breast-mri-lesion-lens/). We provide some utilities to prepare the data. 

## Advanced-MRI-Breast-Lesions datasaet example

To illustrate the provided utilities, we used the case #AMBL-003 from the publicly available [Advanced-MRI-Breast-Lesions datasaet](https://www.cancerimagingarchive.net/collection/advanced-mri-breast-lesions/)[^1]. This case showcases a malignant lesion (invasive ductal carcinoma) in the right breast. 

### Imaging data
Download the DICOM data from the AMBL dataset for case 003. In folder `AIMBL_003/dicom` put a folder containing the DICOM data of DICOM series \#500 with description "500-Registered AX Sen Vibrant MultiPhase". 

From this DICOM data, we will extract the first and last phases in `.mha` format. From the folder `mindful_subream/scripts`, run:

```
python extract_3d_vol_from_4d_dcm.py --dicom_folder ../examples/gc_samples/AMBL_003/dicom --phase_index 0 --output_mha ../examples/gc_samples/AMBL_003/sub_image/AMBL_003_phase_0.mha

python extract_3d_vol_from_4d_dcm.py --dicom_folder ../examples/gc_samples/AMBL_003/dicom --last_phase --output_mha ../examples/gc_samples/AMBL_003/sub_image/AMBL_003_phase_last.mha
```

Now, we will compute the subtracted image:

```
python subtract_images.py --im1 ../examples/gc_samples/AMBL_003/sub_image/AMBL_003_phase_last.mha --im2 ../examples/gc_samples/AMBL_003/sub_image/AMBL_003_phase_0.mha --output ../examples/gc_samples/AMBL_003/sub_image/AMBL_003_sub.mha
```

### Scalar data

Lesion characteristics were computed based on the provided segmentation in the AMBL dataset. It includes a segmentation of the lesion in DICOM SEG format in the DICOM series \#500 with description "ROI". We converted this series to a binary segmentation mask using [3D Slicer](https://www.slicer.org/) with the "Quantitative Reporting" extension installed. For your convenience, the converted file is available as `AMBL_003/lesion_masks/AMBL_003_1_M.mha`.

> [!NOTE]
> File naming convention is important:
> - AMBL_003: a unique ID to identify the case
> - 1: a 1-indexed lesion number, a case can have more than one lesion
> - M: code for "Malignant". Other possibilities are "L" for lymph node, "B" for benign.

In case the segmentation mask was not available, you could create such a binary image with any segmentation software. Note that the mask can have image characteristics that differ from the MRI image (origin, dimensions, etc.). The importance is that the position of its voxels are in the same physical coordinate system as the MRI image.  

Now given this segmentation mask we can compute geometric characteristics:

```
python compute_mask_attributes.py ../examples/gc_samples/AMBL_003/lesion_masks/ --output_csv_file ../examples/gc_samples/AMBL_003/gc/AMBL_003_all_lesions_attributes.csv --classes B:0,M:1,G:2 
```

This will produce the `AMBL_003_all_lesions_attributes.csv` file containing  lesion characteristics. If we had several lesions, we would have to put the corresponding masks with the appropriate naming convention in folder `../examples/gc_samples/AMBL_003/lesion_masks`. 

To be digested by Lesion Lens, the csv file must be converted to configuration `.json` files:

```
python csv_to_lesions_json.py --csv_path ../examples/gc_samples/AMBL_003/gc/AMBL_003_all_lesions_attributes.csv --output_char_json ../examples/gc_samples/AMBL_003/gc/AMBL_003_lesion_information_lesionlens.json --output_pos_json ../examples/gc_samples/AMBL_003/gc/AMBL_003_lesion_positions_lesionlens.json --left_nipple 74.96 -98.49 -28.75 --right_nipple -69.59 -99.65 -44.75
```

The produced files `AMBL_003_lesion_information_lesionlens.json`and `AMBL_003_lesion_positions_lesionlens.json` respectively encode the geometric characteristics of lesion(s) (here only one) with their respective positions in the image. The position of the nipples are optional and are used in the report generation of Lesion Lens. They were manually selected in the MRI image. 

### Categorical data

Categorical data relies on patient information, that  can be found in the `Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx` file of the AMBL dataset.

The Lesion Lens config has the following structure:

```
{
    "age": age at acquisition (integer),
    "BRCA": "negative"/"positive"/"" (string),
    "Chemo": "no"/"yes"/"" (string),
    "FamRisk": "no"/"yes_1st_degree"/"yes_2nd_degree"/"other"/"" (string),
    "PatRisk": "no"/ "yes"/ "other"/"" (string),
    "MenoStatus": "menopause_absent"/ "menopause_present"/ "menopause_withsubstitute"/"perimenopausal_state"/ "" (string),
    "Contraception": "without"/"with"/"other"/ "" (string)
}
```
For example: 

```
{
  "age": 42,
  "BRCA": "negative",
  "Chemo": "no",
  "FamRisk": "no",
  "PatRisk": "no",
  "MenoStatus": "menopause_absent",
  "Contraception": "without"
}
```

Lesion Lens supports partial information, where the empty string "" is given to indicate missing information.

> [!NOTE]
> In the AMBL dataset, relevant information is found in the following columns of the XLSX clinical information file:
> - **age**: <col: age at MRI>
> - **BRCA**: "positive" if <col:reason for referral ID#>=4 or "negative" otherwise, (or info in <col: additional reason for referral ID#>)
> - **Chemo**: "" (no info available)
> - **FamRisk**: "other" if <col:reason for referral ID#>=2 or "no" otherwise (or info in <col: additional reason for referral ID#>)
> - **PatRisk**: "yes" if <col:reason for referral ID#>=3 or "no" otherwise (or info in <col: additional reason for referral ID#>)
> - **MenoStatus**: "" (no info available, could be guessed in some occasions given patient age or gender)
> - **Contraception**: "" (no info available)

For patient 003, no information was available except the patient age (53 years old). The resulting config file is available in `AMBL_003/config`.

### Running the Lesion Lens algorithm

You must register to the Grand-Challenge platform and request the authorization to run the Lesion Lens algorithm. 

The config files for case 003 are available in `AMBL_003/config`, you can check that the output produced by the previous scripts written in the `gc` folder match these config files.

Run the algorithm with "Try-out Algorithm" and upload the produced subtracted image `AMBL_003/sub_image/AMBL_003_sub.mha` along with the config files. See the algorithm page for more information.

The output should match the [public result](https://grand-challenge.org/algorithms/subream-breast-mri-lesion-lens/jobs/d8bcce62-136e-4122-a6d8-fb186e6c0b83) already published online. The produced [report](./AMBL_003/results/AMBL_003_gc_report.pdf) can be also found in the `AMBL_003/results` folder.

[^1]: Daniels, D., Last, D., Cohen, K., Mardor, Y., & Sklair-Levy, M. (2024). Standard and Delayed Contrast-Enhanced MRI of Malignant and Benign Breast Lesions with Histological and Clinical Supporting Data (Advanced-MRI-Breast-Lesions) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/C7X1-YN57
