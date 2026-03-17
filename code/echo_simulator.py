import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from code.constants import TERMINUS, PACKET_LENGTH


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "-n_samples",
        type=int,
        dest="n_samples",
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-f",
        "-fault_type",
        choices=["baseline", "open", "short"],
        dest="fault_type",
        help="Type of fault to simulate",
    )
    parser.add_argument(
        "-o",
        "-output_file",
        type=str,
        dest="output_file",
        help="Output file to save the samples",
    )
    # TODO: make repeatable, but more sensibly than this approach.
    # parser.add_argument(
    #     "-s",
    #     "-seed",
    #     type=int,
    #     default=101,
    #     dest="seed",
    #     help="Seed for the random number generator",
    # )
    parser.add_argument(
        "-p", "-plot", action="store_true", dest="plot", help="Plot the samples"
    )

    return parser.parse_args()


def baseline_echo(n_samples: int, packet_peak: float = TERMINUS * 0.8) -> pd.DataFrame:
    baseline = norm.pdf(
        np.linspace(0, TERMINUS, PACKET_LENGTH), loc=packet_peak, scale=TERMINUS / 25
    )
    baseline = baseline / np.max(baseline)

    noise_sd = 1e-1
    samples = np.tile(baseline, [n_samples, 1]) + np.random.normal(
        0, noise_sd, [n_samples, PACKET_LENGTH]
    )
    return pd.DataFrame(samples)


def open_fault_echo(n_samples: int) -> pd.DataFrame:
    packet_peak = TERMINUS * np.random.uniform(0.7, 0.9)

    samples = baseline_echo(n_samples, packet_peak)
    return pd.DataFrame(samples)


def short_fault_echo(n_samples: int) -> pd.DataFrame:
    packet_peak = TERMINUS * np.random.uniform(0.1, 0.7)

    samples = -1 * baseline_echo(n_samples, packet_peak)
    return pd.DataFrame(samples)


def generate_samples(n_samples: int, fault_type: str) -> pd.DataFrame:

    match fault_type:
        case "baseline":
            return baseline_echo(n_samples)
        case "open":
            return open_fault_echo(n_samples)
        case "short":
            return short_fault_echo(n_samples)
        case _:
            raise ValueError(f"Invalid fault type: {fault_type}")


def save_samples(samples: pd.DataFrame, output_file_path: str):
    samples.to_csv(output_file_path, header=False, index=False)


def plot_samples(samples: pd.DataFrame, fault_type: str):

    for _, row in samples.iterrows():
        plt.plot(row, alpha=0.5)

    plt.ylim(-1.1, 1.1)
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s E-7)")
    plt.title(f"Echo Signal: {fault_type.capitalize()}")
    plt.show()


def main():
    args = parse_arguments()
    n_samples = args.n_samples
    output_file = args.output_file
    fault_type = args.fault_type

    samples = generate_samples(n_samples, fault_type)

    if args.plot:
        plot_samples(samples, fault_type)

    save_samples(samples, output_file)


if __name__ == "__main__":
    main()
