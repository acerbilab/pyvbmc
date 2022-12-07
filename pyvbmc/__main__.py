from pathlib import Path
from os import system

if __name__ == "__main__":
    # Launch Jupyter Notebook examples with python -m pyvbmc

    # binary/wheel install:
    base_path = Path(__file__).parent
    examples_path = base_path.joinpath("examples")
    command = f"jupyter notebook {examples_path}"
    exit_status = system(command)

    if exit_status != 0:
        # If failed, look for examples directory one level up
        # (source install):
        base_path = Path(__file__).parent.parent
        examples_path = base_path.joinpath("examples")
        command = f"jupyter notebook {examples_path}"
        system(command)
