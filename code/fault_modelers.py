from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from baseline_modelers import MahalanobisBaselineModel

from constants import FaultType


class FaultClassifier(ABC):
    @abstractmethod
    def classify(self, samples: pd.DataFrame) -> FaultType:
        pass


class MahalanobisFaultClassifier(FaultClassifier):
    """
    Classify faults based on the peak voltage relative the baseline peak voltage.

    Leverage business logic to classsify into one of two hard fault types: open or short.
    """

    def __init__(self, baseline_model: MahalanobisBaselineModel):
        self.baseline_model = baseline_model

    def classify(self, samples: pd.DataFrame) -> FaultType:

        _ = self.baseline_model.predict(samples)

        peak_idx = int(np.argmax(np.abs(self.baseline_model.sample_mean)))
        peak_voltage = self.baseline_model.sample_mean[peak_idx]

        # Check for hard faults
        if peak_voltage < 0:
            return FaultType.SHORT
        elif peak_idx < self.baseline_model.peak_voltage:
            return FaultType.OPEN
        else:
            return FaultType.UNKNOWN
