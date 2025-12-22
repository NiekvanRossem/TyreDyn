from src.utils.misc import allowableData
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
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Returns the lateral relaxation length.

        Parameters
        ----------
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : str, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        sigma_y : allowableData
            Lateral relaxation length.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, IA = self._format_check([FZ, IA])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # cornering stiffness
        KYA = self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # lateral stiffness
        Cy = self.stiffness.find_lateral_stiffness(FZ, P)

        # lateral relaxation length (A3.9)
        sigma_y = KYA / Cy
        return sigma_y

    def find_longitudinal_relaxation(
            self,
            FZ: allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Returns the longitudinal relaxation length.

        Parameters
        ----------
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).

        Returns
        -------
        sigma_x : allowableData
            Longitudinal relaxation length.
        """
        
        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ = self._format_check(FZ)

        # slip stiffness
        KXK = self.gradient.find_slip_stiffness(FZ, P)

        # longitudinal stiffness
        Cx = self.stiffness.find_longitudinal_stiffness(FZ, P)

        # longitudinal relaxation length (A3.9)
        sigma_x = KXK / Cx
        return sigma_x
