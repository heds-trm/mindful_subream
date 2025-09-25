import argparse

from mindful_subream.dataset.dataset_master_builder import SubreamMasterBuilder


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config", type=str)
    args = arg_parser.parse_args()

    subream_master_builder = SubreamMasterBuilder(args.config)
    subream_master_builder.run()


if __name__ == "__main__":
    main()
