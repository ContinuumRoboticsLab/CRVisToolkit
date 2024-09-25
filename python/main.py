import sys
import os

from plotter.main import plot_from_file
from parser import parser

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    args = parser.parse_args()
    
    match args.task:
        case "plotter":
            plot_from_file(args.file_path)
        case "IK":
            raise NotImplementedError("IK task is not implemented yet.")
        case _:
            print("Invalid task. Please specify a valid task.")
            exit