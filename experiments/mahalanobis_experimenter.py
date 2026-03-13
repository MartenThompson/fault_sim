import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from code.constants import PACKET_LENGTH, IsFault
from code.baseline_modelers import MahalanobisBaselineModel
from code.echo_simulator import baseline_echo, open_fault_echo, save_samples


def parse_arguments():
    parser = argparse.ArgumentParser(
        usage="python3.12 -m experiments.mahalanobis_experimenter -d data -e open_fault"
    )

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
        choices=["open_fault", "short_fault"],
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


def test_error_rates_open_fault(
    experiment_dir: str,
    n_training_samples: int,
    n_test_samples: int,
    n_experiments: int,
):
    # For now, test samples will be split evenly between no fault and open fault.
    training_dir, test_dir, results_dir = create_experiment_dirs(experiment_dir)

    false_positive = 0
    true_negative = 0
    true_positive = 0
    false_negative = 0

    for i in range(n_experiments):
        baseline_samples = baseline_echo(n_samples=n_training_samples)
        save_samples(baseline_samples, os.path.join(training_dir, f"baseline_{i}.csv"))
        baseline_model = MahalanobisBaselineModel(
            packet_length=PACKET_LENGTH, significance_threshold=17.0
        )
        baseline_model.train(baseline_samples)
        baseline_model.fit()

        no_fault_samples = baseline_echo(n_samples=n_test_samples // 2)
        save_samples(no_fault_samples, os.path.join(test_dir, f"no_fault_{i}.csv"))

        for _, sample in no_fault_samples.iterrows():
            prediction = baseline_model.predict(sample)
            if prediction == IsFault.FAULT:
                false_positive += 1
            else:
                true_negative += 1

        open_fault_samples = open_fault_echo(n_samples=n_test_samples // 2)
        save_samples(open_fault_samples, os.path.join(test_dir, f"open_fault_{i}.csv"))

        for _, sample in open_fault_samples.iterrows():
            prediction = baseline_model.predict(sample)
            if prediction == IsFault.FAULT:
                true_positive += 1
            else:
                false_negative += 1

    true_positive_rate = true_positive / (n_experiments * n_test_samples // 2)
    false_negative_rate = false_negative / (n_experiments * n_test_samples // 2)
    true_negative_rate = true_negative / (n_experiments * n_test_samples // 2)
    false_positive_rate = false_positive / (n_experiments * n_test_samples // 2)

    results = {
        "n_training_samples": n_training_samples,
        "n_test_samples": n_test_samples,
        "n_experiments": n_experiments,
        "true_positive_rate": true_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
    }

    fig, ax = plt.subplots()
    cm = np.array([[true_positive, false_positive], [false_negative, true_negative]])
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Fault", "No Fault"],
        yticklabels=["Fault", "No Fault"],
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))

    results_path = os.path.join(results_dir, "open_fault_results.csv")
    print(f"Saving results to {results_path}")

    pd.DataFrame(results, index=[0]).to_csv(results_path, index=False)


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
        case "open_fault":
            test_error_rates_open_fault(experiment_dir, 1000, 10, 100)
        case "short_fault":
            test_error_rates_short_fault(experiment_dir)
        case _:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

    pass


if __name__ == "__main__":
    main()
