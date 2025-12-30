from src.utils.formatting import SignalLike, AngleUnit
from typing import Literal
import numpy as np

class RelaxationMF61:
    """
    Relaxation length module for MF 6.1.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # other modules
        self.stiffness  = model.stiffness
        self.gradient   = model.gradient

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_lateral_relaxation(
            self,
            *,
            FZ:   SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            PHIT: SignalLike = 0.0,
            angle_unit: AngleUnit = "rad"
    ) -> SignalLike:
        """
        Returns the lateral relaxation length.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        sigma_y : SignalLike
            Lateral relaxation length.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA, PHIT = self._format_check([FZ, P, IA, PHIT])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # cornering stiffness
        KYA = self.gradient.find_cornering_stiffness(FZ=FZ, P=P, IA=IA, PHIT=PHIT, angle_unit=angle_unit)

        # lateral stiffness
        Cy = self.stiffness.find_lateral_stiffness(FZ=FZ, P=P)

        # lateral relaxation length (A3.9)
        sigma_y = KYA / Cy
        return sigma_y

    def find_longitudinal_relaxation(
            self,
            *,
            FZ: SignalLike,
            P:  SignalLike = None
    ) -> SignalLike:
        """
        Returns the longitudinal relaxation length.

        Parameters
        ----------
        FZ : SignalLike
            Vertical load.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        sigma_x : SignalLike
            Longitudinal relaxation length.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # slip stiffness
        KXK = self.gradient.find_slip_stiffness(FZ=FZ, P=P)

        # longitudinal stiffness
        Cx = self.stiffness.find_longitudinal_stiffness(FZ=FZ, P=P)

        # longitudinal relaxation length (A3.9)
        sigma_x = KXK / Cx
        return sigma_x
