from enum import Enum

PACKET_LENGTH = 100
TERMINUS = 1e-3


# TODO: work into arg parser
class FaultType(Enum):
    BASELINE = "baseline"
    OPEN = "open"
    SHORT = "short"
    UNKNOWN = "unknown"
