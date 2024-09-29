import unittest


def run_all_tests(test_directory=".", pattern="test*.py", verbosity=2):
    """
    Run all unittest test cases in the specified directory and its subdirectories.

    Args:
        test_directory (str): The directory to start looking for test cases.
                              Defaults to the current directory.
    """
    # Create a test loader
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_directory, pattern="test*.py")
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return the result to indicate success or failure
    return result.wasSuccessful()


# Run the function
if __name__ == "__main__":
    success = run_all_tests("path/to/your/tests")
    if not success:
        exit(1)
