This folder contains the `balanced_lymphnodes` folds for all existing modes (ie. `b_g_m`, `b_m`, ...).
This means that lymphondes were randomly removed from folds so that their number matches the number of benin lesions.

As with default folds, these folds were randomly generated, with two contraints:
	1) Different images from a single patient are in the same partitions.
	2) The selected configuration is the one for which the prevalence of each label across partitions is the closest.