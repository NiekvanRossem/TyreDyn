from src.utils.misc import allowableData
import numpy as np
from typing import Literal

class GradientsMF61:
    """
    Module containing the gradient functions such as cornering stiffness or camber stiffness for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

        self.turn_slip  = model.turn_slip

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_cornering_stiffness(
            self,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Finds the cornering stiffness at zero slip angle for pure slip conditions.

        :param PHI:
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``KYA`` -- cornering stiffness.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_3 = self.turn_slip.__find_zeta_3(PHI)
        else:
            zeta_3 = self.zeta_3_default

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # unpack tyre properties
        FZ0 = self.FNOMIN

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA = self._format_check([FZ, P, IA])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # corrected camber angle
        gamma_star = self.correction.__find_gamma_star(IA)

        # normalize pressure
        dpi = self.normalize.__find_dpi(P)

        # scaled nominal load
        FZ0_prime = FZ0 * self.LFZO

        # cornering stiffness (4.E25)
        KYA = (self.PKY1 * FZ0_prime * (1.0 + self.PPY1 * dpi) * (1.0 - self.PKY3 * np.abs(gamma_star))
               * self.sin(self.PKY4 * np.atan2(FZ / FZ0_prime, (self.PKY2 + self.PKY5 * gamma_star ** 2)
                                             * (1.0 + self.PPY2 * dpi)))) * zeta_3 * self.LKY
        return KYA

    def find_slip_stiffness(self, FZ: allowableData, P:  allowableData = None) -> allowableData:
        """
        Finds the longitudinal slip stiffness at zero slip ratio for pure slip conditions.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``KXK`` -- longitudinal slip stiffness.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # normalize pressure and load
        dfz = self.normalize.__find_dfz(FZ)
        dpi = self.normalize.__find_dpi(P)

        # slip stiffness (4.E15)
        KXK = (FZ * (self.PKX1 + self.PKX2 * dfz) * np.exp(self.PKX3 * dfz)
               * (1.0 + self.PPX1 * dpi + self.PPX2 * dpi ** 2) * self.LKX)
        return KXK

    def find_camber_stiffness(self, FZ: allowableData, P:  allowableData = None) -> allowableData:
        """
        Finds the camber stiffness. Calculations according to  Pacejka's MF 6.1.2 model.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``KYCO`` -- camber stiffness.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = self._format_check([FZ, P])

        # normalize pressure and load
        dfz = self.normalize.__find_dfz(FZ)
        dpi = self.normalize.__find_dpi(P)

        # camber stiffness (4.E30)
        KYCO = FZ * (self.PKY6 + self.PKY7 * dfz) * (1.0 - self.PPY5 * dpi) * self.LKYC
        return KYCO

    @staticmethod
    def find_instant_kya(SA: allowableData, FY: allowableData) -> allowableData:
        """
        Finds the instantaneous cornering stiffness of the tyre by calculating the gradient of the lateral force and the
        slip ratio.

        :param SA:
        :param FY:
        :return:
        """

        # instantaneous cornering stiffness (from MFeval)
        iKYA = np.gradient(FY, SA)
        return iKYA

    @staticmethod
    def find_instant_kxk(SL: allowableData, FX: allowableData) -> allowableData:
        """
        Finds the instantaneous slip stiffness of the tyre by calculating the gradient of the longitudinal force and the
        slip ratio.

        :param SL: slip ratio.
        :param FX: longitudinal force.

        :return: ``iKXK`` -- instantaneous slip stiffness.
        """

        # instantaneous slip stiffness from MFeval)
        iKXK = np.gradient(FX, SL)
        return iKXK
