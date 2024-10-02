import argparse

parser = argparse.ArgumentParser(description="Parser for running the plotter/IK")

parser.add_argument("task", type=str, help="Task to run [plotter/IK]")
parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=False,
    help="Filepath to the target data file",
)
parser.add_argument(
    "-e", "--example", type=str, required=False, help="Example to run [ctcr/tdcr]"
)
