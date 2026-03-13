import argparse
from datetime import datetime
import os

from echo_simulator import baseline_echo, save_samples


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "-experiement_dir",
        type=str,
        dest="experiment_dir",
        help="Directory to save the experiment results",
    )

    parser.add_argument(
        "-e",
        "-experiment_type",
        choices=["no_fault", "open_fault", "short_fault"],
        dest="experiment_type",
        help="Type of experiment to run",
    )

    return parser.parse_args()


def create_experiment_dirs(experiment_dir: str) -> tuple[str, str, str]:
    training_dir = os.path.join(experiment_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    test_dir = os.path.join(experiment_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return training_dir, test_dir, results_dir


def test_error_rates_no_fault(experiment_dir: str):
    training_dir, test_dir, results_dir = create_experiment_dirs(experiment_dir)

    # Generate training data.
    baseline_samples = baseline_echo(n_samples=1000, seed=101)
    save_samples(baseline_samples, os.path.join(training_dir, "baseline.csv"))

    # Train baseline model.

    # Generate test data: no fault.

    # Test baseline model.

    # Calculate true/false positive/negative rates.

    pass


def test_error_rates_open_fault():
    # Make dirs for training data.

    # Generate training data.

    # Train baseline model.

    # Generate test data: open fault.

    # Test baseline model.

    # Calculate true/false positive/negative rates.

    pass


def test_error_rates_short_fault():
    # Make dirs for training data.

    # Generate training data.

    # Train baseline model.

    # Generate test data: short fault.

    # Test baseline model.

    # Calculate true/false positive/negative rates.

    pass


def main():
    args = parse_arguments()

    experiment_type = args.experiment_type
    experiment_dir_root = args.experiment_dir

    os.makedirs(experiment_dir_root, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiment_dir_root, f"{experiment_type}_{now}")
    os.makedirs(experiment_dir, exist_ok=True)

    match experiment_type:
        case "no_fault":
            test_error_rates_no_fault(experiment_dir)
        case "open_fault":
            test_error_rates_open_fault(experiment_dir)
        case "short_fault":
            test_error_rates_short_fault(experiment_dir)
        case _:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

    pass


if __name__ == "__main__":
    main()
