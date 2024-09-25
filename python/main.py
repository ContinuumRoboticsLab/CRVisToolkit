from plotter import ctcr, tdcr
from parser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    
    match args.task:
        case "plotc":
            ctcr.plot_from_file(args.file_path)
        case "plott":
            tdcr.plot_from_file(args.file_path)
        case "IK":
            raise NotImplementedError("IK task is not implemented yet.")
        case _:
            print("Invalid task. Please specify a valid task.")
            exit