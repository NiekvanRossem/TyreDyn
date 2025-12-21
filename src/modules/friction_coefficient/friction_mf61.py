from src.utils.misc import allowableData
from src.helpers.corrections import CorrectionsMF61
from src.helpers.normalize import Normalize
from typing import Literal

class FrictionMF61:
    """
    Friction coefficient module for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model     = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_mu_x(
            self,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Finds the longitudinal friction coefficient.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``mu_x`` -- longitudinal friction coefficient.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA, VS = self._format_check([FZ, P, IA, VS])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # unpack tyre properties
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.correction._find_lmu_star(VS, V0, self.LMUX)

        # friction coefficient (4.E13)
        mu_x = ((self.PDX1 + self.PDX2 * dfz) * (1.0 + self.PPX3 * dpi + self.PPX4 * dpi ** 2)
                * (1.0 - self.PDX3 * IA ** 2) * LMUX_star)

        return mu_x

    def find_mu_y(
            self,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Finds the lateral friction coefficient.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``mu_y`` -- lateral friction coefficient.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA, VS = self._format_check([FZ, P, IA, VS])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # unpack tyre properties
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # composite friction scaling factor (4.E7)
        LMUY_star = self.correction._find_lmu_star(VS, V0, self.LMUY)

        # lateral friction coefficient (4.E23)
        mu_y = ((self.PDY1 + self.PDY2 * dfz) * (1.0 + self.PPY3 * dpi + self.PPY4 * dpi ** 2)
                * (1.0 - self.PDY3 * gamma_star ** 2) * LMUY_star)

        return mu_y
