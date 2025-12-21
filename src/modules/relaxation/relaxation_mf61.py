from src.utils.misc import allowableData
from typing import Literal
import numpy as np

class RelaxationMF61:
    """
    Relaxation length module for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
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
        Finds the lateral relaxation length.

        :param PHI:
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``sigma_y`` -- lateral relaxation length.
        """

        # set default value for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA = self._format_check([FZ, P, IA])

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
        Finds the longitudinal relaxation length.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``sigma_x`` -- longitudinal relaxation length.
        """

        # set default value for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # slip stiffness
        KXK = self.gradient.find_slip_stiffness(FZ, P)

        # longitudinal stiffness
        Cx = self.stiffness.find_longitudinal_stiffness(FZ, P)

        # longitudinal relaxation length (A3.9)
        sigma_x = KXK / Cx
        return sigma_x
