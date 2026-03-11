import numpy as np
import pandas as pd
from baseline_modelers import BaselineModel

from constants import FaultType


class FaultDetector:
    def __init__(self, baseline_model: BaselineModel):
        self.baseline_model = baseline_model

    # TODO: enable batch analysis
    def analyze(self, single_sample: pd.Series):
        return self.baseline_model.predict(single_sample)


class FaultClassifier:
    def __init__(self, baseline_model: BaselineModel):
        self.baseline_model = baseline_model

    def classify(self, samples: pd.DataFrame):

        score, residual = self.baseline_model.predict(samples)

        abs_residual = np.abs(residual)
        peak_idx = int(np.argmax(abs_residual))
        peak_voltage = residual[peak_idx]

        # Check for hard faults
        if peak_voltage < 0:
            return FaultType.SHORT
        elif peak_idx < self.baseline_model.peak_voltage:
            return FaultType.OPEN
        else:
            return FaultType.UNKNOWN
