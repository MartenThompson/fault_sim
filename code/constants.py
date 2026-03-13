from enum import Enum

PACKET_LENGTH = 100
TERMINUS = 1e-3


class IsFault(Enum):
    FAULT = "fault"
    NOT_FAULT = "not_fault"


# TODO: work into arg parser
class FaultType(Enum):
    OPEN = "open"
    SHORT = "short"
    UNKNOWN = "unknown"
