from plotter import ctcr, tdcr
from tests import run_all_tests
from examples import plot_example


from parser import parser

if __name__ == "__main__":
    args = parser.parse_args()

    match args.task:
        case "plotc":
            ctcr.plot_from_file(args.file_path)
        case "plott":
            tdcr.plot_from_file(args.file_path)
        case "example":
            sample_task = args.example
            plot_example(sample_task)
        case "IK":
            raise NotImplementedError("IK task is not implemented yet.")
        case "test":
            run_all_tests()
        case _:
            print("Invalid task. Please specify a valid task.")
            exit(1)
