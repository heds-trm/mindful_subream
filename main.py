from mindful_core.main import main as core_main
from mindful_core.experiments.experiment_round import ExperimentRound

from mindful_subream.models.classification.sieve import MultimodalSieve
from mindful_subream.models.classification.ultra import UlTra
import mindful_subream


def main():
    ExperimentRound.archived_modules.append(mindful_subream)
    core_main()


if __name__ == "__main__":
    main()
