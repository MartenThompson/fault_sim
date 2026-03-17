import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from code.constants import PACKET_LENGTH, IsFault
from code.baseline_modelers import MahalanobisBaselineModel
from code.echo_simulator import baseline_echo, open_fault_echo


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


def plot_confusion_matrix(results_path: str):
    # TODO: test
    results_dir = os.path.dirname(results_path)
    results_df = pd.read_csv(results_path)
    true_positive = results_df["true_positive_rate"].sum()
    false_positive = results_df["false_positive_rate"].sum()
    false_negative = results_df["false_negative_rate"].sum()
    true_negative = results_df["true_negative_rate"].sum()
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
    plt.close()


def roc_curve(results_path: str):
    results_dir = os.path.dirname(results_path)
    results_df = pd.read_csv(results_path)
    grouped = results_df.groupby("significance_threshold")

    summary = grouped.agg(
        false_positive_mean=("false_positive_rate", "mean"),
        true_positive_mean=("true_positive_rate", "mean"),
        true_positive_q05=("true_positive_rate", lambda x: x.quantile(0.1)),
        true_positive_q95=("true_positive_rate", lambda x: x.quantile(0.9)),
    ).reset_index()

    # Sort by mean FPR so the curve is well-ordered on the x-axis.
    summary = summary.sort_values("false_positive_mean")

    fpr = summary["false_positive_mean"].values
    tpr_mean = summary["true_positive_mean"].values
    tpr_q05 = summary["true_positive_q05"].values
    tpr_q95 = summary["true_positive_q95"].values

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr_mean, label="Mean")
    ax.fill_between(fpr, tpr_q05, tpr_q95, alpha=0.2, label="TPR 10-90%")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_box_aspect(1)
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()


def precision_recall_curve(results_path: str):
    results_dir = os.path.dirname(results_path)
    results_df = pd.read_csv(results_path)

    precision = results_df["true_positive_rate"] / (
        results_df["true_positive_rate"] + results_df["false_positive_rate"]
    )
    recall = results_df["true_positive_rate"] / (
        results_df["true_positive_rate"] + results_df["false_negative_rate"]
    )
    results_df = results_df.assign(precision=precision, recall=recall)

    grouped = results_df.groupby("significance_threshold")
    summary = grouped.agg(
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        precision_q05=("precision", lambda x: x.quantile(0.05)),
        precision_q95=("precision", lambda x: x.quantile(0.95)),
    ).reset_index()

    # Sort by mean recall so the curve is ordered along the x-axis.
    summary = summary.sort_values("recall_mean")

    recall_mean = summary["recall_mean"].values
    precision_mean = summary["precision_mean"].values
    precision_q05 = summary["precision_q05"].values
    precision_q95 = summary["precision_q95"].values

    fig, ax = plt.subplots()
    ax.plot(recall_mean, precision_mean, label="Mean")
    ax.fill_between(recall_mean, precision_q05, precision_q95, alpha=0.2, label="5-95%")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_box_aspect(1)
    plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
    plt.close()


def test_error_rates_open_fault(
    experiment_dir: str,
    n_training_samples: int,
    n_test_samples: int,
    significance_thresholds: np.ndarray,
) -> pd.DataFrame:
    # For now, test samples will be split evenly between no fault and open fault.
    training_dir, test_dir, results_dir = create_experiment_dirs(experiment_dir)

    baseline_samples = baseline_echo(n_samples=n_training_samples)
    no_fault_samples = baseline_echo(n_samples=n_test_samples // 2)
    open_fault_samples = open_fault_echo(n_samples=n_test_samples // 2)
    results_all = np.empty(len(significance_thresholds), dtype=pd.DataFrame)

    for i, significance_threshold in enumerate(significance_thresholds):
        baseline_model = MahalanobisBaselineModel(
            packet_length=PACKET_LENGTH, significance_threshold=significance_threshold
        )
        baseline_model.train(baseline_samples)
        baseline_model.fit()

        false_positive = 0
        true_negative = 0
        true_positive = 0
        false_negative = 0

        for _, sample in no_fault_samples.iterrows():
            prediction = baseline_model.predict(sample)
            if prediction == IsFault.FAULT:
                false_positive += 1
            else:
                true_negative += 1

        for _, sample in open_fault_samples.iterrows():
            prediction = baseline_model.predict(sample)
            if prediction == IsFault.FAULT:
                true_positive += 1
            else:
                false_negative += 1

        true_positive_rate = true_positive / (n_test_samples // 2)
        false_negative_rate = false_negative / (n_test_samples // 2)
        true_negative_rate = true_negative / (n_test_samples // 2)
        false_positive_rate = false_positive / (n_test_samples // 2)

        results = {
            "n_training_samples": n_training_samples,
            "n_test_samples": n_test_samples,
            "significance_threshold": significance_threshold,
            "true_positive_rate": true_positive_rate,
            "false_negative_rate": false_negative_rate,
            "true_negative_rate": true_negative_rate,
            "false_positive_rate": false_positive_rate,
        }
        results_all[i] = pd.DataFrame(results, index=[0])

    results_df = pd.concat(results_all)
    return results_df


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

    n_experiments = 500
    significance_thresholds = np.linspace(1.0, 20.0, 20)
    experiment_type = args.experiment_type
    experiment_dir_root = args.experiment_dir

    os.makedirs(experiment_dir_root, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiment_dir_root, f"{experiment_type}_{now}")
    os.makedirs(experiment_dir, exist_ok=True)

    match experiment_type:
        case "open_fault":
            results_dfs = np.empty(
                n_experiments * len(significance_thresholds), dtype=pd.DataFrame
            )
            for i in range(n_experiments):
                results_dfs[i] = test_error_rates_open_fault(
                    experiment_dir, 1000, 100, significance_thresholds
                )
            results_df = pd.concat(results_dfs)
            results_path = os.path.join(
                os.path.join(experiment_dir, "results"), "open_fault_results.csv"
            )
            results_df.to_csv(results_path, index=False)
            roc_curve(results_path)
            precision_recall_curve(results_path)
        case "short_fault":
            test_error_rates_short_fault(experiment_dir)
        case _:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

    pass


if __name__ == "__main__":
    main()
