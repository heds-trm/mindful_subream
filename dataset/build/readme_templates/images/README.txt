This folder contains all image data for this build.
Currently, this includes:
	- {roi_size}mm ROIs for:
		1) All 13 sub-phases (substraction between phase 0 and phase `n`.) Each phase is stored in a separate file.
		2) MIP computed along axis n°{mip_dim}. All phases are contained in one file. All samples were augmented {mip_augmentation_count} times.