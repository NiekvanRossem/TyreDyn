from dataclasses import dataclass
from tyredyn.types.aliases import SignalLike

@dataclass(frozen=True)
class InputSignals:
    """Stores the input signals in a dataclass."""
    SA:   SignalLike # slip angle
    SL:   SignalLike # slip ratio
    FZ:   SignalLike # vertical force
    N:    SignalLike # rotational speed of the wheel
    P:    SignalLike # tyre pressure
    IA:   SignalLike # tyre inclination angle (with respect to the ground plane)
    VX:   SignalLike # contact patch longitudinal speed
    PHIT: SignalLike # turn slip