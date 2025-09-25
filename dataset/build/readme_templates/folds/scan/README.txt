This folder contains the default folds for all existing modes (ie. `b_g_m`, `b_m`, ...).
Several sub-versions exists, such as the `default` one, or the `balanced_lymphnodes` one where the number of lymphnodes was balanced against the number of benin lesions.

These folds were randomly generated, with two contraints:
	1) Different images from a single patient are in the same partitions.
	2) The selected configuration is the one for which the prevalence of each label across partitions is the closest.