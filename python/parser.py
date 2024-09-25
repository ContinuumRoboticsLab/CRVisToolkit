import argparse

parser = argparse.ArgumentParser(description="Parser for running the plotter/IK")

parser.add_argument("task", type=str, help="Task to run [plotter/IK]")
parser.add_argument("file_path", type=str, help="Filepath to the target data file")