import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
        choices=["baseline"],
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
    parser.add_argument(
        "-s",
        "-seed",
        type=int,
        default=101,
        dest="seed",
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "-p", "-plot", action="store_true", dest="plot", help="Plot the samples"
    )

    return parser.parse_args()


def baseline_echo(n_samples: int) -> pd.DataFrame:
    samples = np.tile(np.repeat(0, 100), [n_samples, 1])
    return pd.DataFrame(samples)


def generate_samples(n_samples: int, fault_type: str, seed: int) -> pd.DataFrame:
    if fault_type == "baseline":
        return baseline_echo(n_samples)
    else:
        raise ValueError(f"Invalid fault type: {fault_type}")


def save_samples(samples: pd.DataFrame, output_file_path: str):
    samples.to_csv(output_file_path, header=False, index=False)


def plot_samples(samples: pd.DataFrame):

    for _, row in samples.iterrows():
        plt.plot(row)
    plt.show()


def main():
    args = parse_arguments()
    n_samples = args.n_samples
    output_file = args.output_file
    fault_type = args.fault_type
    seed = args.seed

    samples = generate_samples(n_samples, fault_type, seed)

    if args.plot:
        plot_samples(samples)

    save_samples(samples, output_file)


if __name__ == "__main__":
    main()
