from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaselineModel(ABC):
    @abstractmethod
    def train(self, burn_in_samples: pd.DataFrame):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, samples: pd.DataFrame):
        pass


class MahalanobisBaselineModel(BaselineModel):
    def __init__(self, packet_length: int, regularization: float = 1e-6):
        self.packet_length = packet_length
        self.regularization = regularization

    def train(self, burn_in_samples: pd.DataFrame) -> None:

        assert burn_in_samples.shape[1] == self.packet_length, (
            f"Burn-in samples must have length {self.packet_length}. Instead got {burn_in_samples.shape[1]}"
        )
        assert burn_in_samples.shape[0] > 0, (
            "Burn-in samples must have at least one sample"
        )

        self.burn_in_samples = burn_in_samples.values
        self.n_samples = burn_in_samples.shape[0]

    def fit(self) -> None:
        self.mean = np.mean(self.burn_in_samples, axis=0)
        self.cov = np.cov(self.burn_in_samples.T) + self.regularization * np.eye(
            self.packet_length
        )
        self.precision = np.linalg.inv(self.cov)

        self.peak_idx = int(np.argmax(np.abs(self.mean)))
        self.peak_voltage = self.mean[self.peak_idx]

    def predict(self, samples: pd.DataFrame) -> tuple[float, np.ndarray]:
        """
        Compute the Mahalanobis distance between new samples and the burn-in samples.

        D = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
        """
        assert samples.shape[1] == self.packet_length, (
            f"Samples must have length {self.packet_length}. Instead got {samples.shape[1]}"
        )
        assert samples.shape[0] > 0, "Samples must have at least one sample"

        samples = samples.values
        mean = self.mean
        precision = self.precision

        residual = np.sum(samples - mean, axis=1)
        D_squared = residual @ precision * residual
        D = np.sqrt(np.max(D_squared, 0.0))

        return D, residual
