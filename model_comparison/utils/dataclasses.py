from dataclasses import dataclass
from typing import Literal
from tyredyn.types.aliases import SignalLike

@dataclass()
class OutputSignals:
    """
    Dataclass containing the full state output signals. Applicable to both TyreDyn and MFeval. Signals that are specific
    to TyreDyn are set to optional (default value is None).
    """

    # output signals
    FX:         SignalLike
    FY:         SignalLike
    FZ:         SignalLike
    MX:         SignalLike
    MY:         SignalLike
    MZ:         SignalLike
    SL:         SignalLike
    SA:         SignalLike
    IA:         SignalLike
    PHIT:       SignalLike
    VX:         SignalLike
    P:          SignalLike
    N:          SignalLike
    RE:         SignalLike
    rho:        SignalLike
    RL:         SignalLike
    a:          SignalLike
    b:          SignalLike
    t:          SignalLike
    mu_x:       SignalLike
    mu_y:       SignalLike
    MZR:        SignalLike
    Cx:         SignalLike
    Cy:         SignalLike
    Cz:         SignalLike
    KYA:        SignalLike
    iKYA:       SignalLike
    KXK:        SignalLike
    sigma_x:    SignalLike
    sigma_y:    SignalLike

    # TyreDyn-specific signals
    R_omega:    SignalLike = None
    iKXK:       SignalLike = None

    # unit system
    units: Literal["SI", "display"] = "SI"