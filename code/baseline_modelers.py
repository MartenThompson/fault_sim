from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from code.constants import IsFault


class BaselineModel(ABC):
    """
    Baseline models are used to fingerprint normally functioning signals and make the binary classification of new samples as either faulted or not faulted.

    They are used in three phases:
    1. Training: add burn-in samples of normally functioning signals.
    2. Fitting: compute representative statistics/models.
    3. Predicting: classify new samples as faulted or not.
    """

    @abstractmethod
    def train(self, burn_in_samples: pd.DataFrame):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, samples: pd.DataFrame) -> IsFault:
        pass


class MahalanobisBaselineModel(BaselineModel):
    """
    The Mahalanobis distance generalizes the (squared) z-score to multivariate data. Its test statistic is defined as:

    D^2 = (x - mu)^T * Sigma^-1 * (x - mu)
    where x is the sample, mu is the mean, and Sigma is the covariance matrix.

    For normally distributed data, the Mahalanobis distance follows a chi-squared distribution with degrees of freedom equal to the packet_length.

    It is sensitive to outliers in the training data; future work will incorporate more robust covariance estimation techniques.
    """

    def __init__(
        self,
        packet_length: int,
        significance_threshold: float,
        regularization: float = 1e-6,
    ):
        self.is_fit = False
        self.packet_length = packet_length
        self.significance_threshold = significance_threshold
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
        self.is_fit = True

    def predict(self, sample: pd.Series) -> IsFault:
        """
        Compute the Mahalanobis distance between new sample (single) and the burn-in samples.

        D = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
        """
        assert self.is_fit, "Model must be fit before predicting"

        assert len(sample) == self.packet_length, (
            f"Sample must have length {self.packet_length}. Instead got {len(sample)}"
        )

        sample = sample.values
        self.sample = sample

        mean = self.mean
        precision = self.precision
        # residual = np.sum(sample - mean, axis=1)
        residual = sample - mean
        D_squared = residual @ precision @ residual

        D = np.sqrt(np.max([D_squared, 0.0]))

        self.residual = residual
        self.D = D

        if D > self.significance_threshold:
            return IsFault.FAULT
        else:
            return IsFault.NOT_FAULT
