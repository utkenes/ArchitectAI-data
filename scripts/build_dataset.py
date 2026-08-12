"""
Script to execute full dataset build pipeline
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from click.testing import CliRunner
from architectai_dataset_builder.cli import build_dataset


def main():
    runner = CliRunner()
    result = runner.invoke(build_dataset, ["--build-id", "build_v1_run1"])
    print(result.output)
    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
