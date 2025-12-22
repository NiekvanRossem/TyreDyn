from src.utils.misc import allowableData
from typing import Literal
import numpy as np

class TrailMF61:
    """
    Pneumatic trail module for MF 6.1.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other modules
        self.turn_slip  = model.turn_slip

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_trail_pure(
            self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Finds the pneumatic trail of the tyre for pure slip conditions.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Slip speed magnitude (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        t : allowableData
            Pneumatic trail.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX
        PHI = 0.0 if PHI is None else PHI

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VC, VCX, VS = self._format_check([SA, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            zeta_5 = self.turn_slip._find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_default

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC, VCX)

        # find coefficients
        BT, CT, DT, ET, alpha_t = self.__trail_main_routine(SA, FZ, P, IA, VCX, VS, zeta_5)

        # pneumatic trail (4.E33)
        t = DT * np.cos(CT * self.atan(BT * alpha_t - ET * (BT * alpha_t - self.atan(BT * alpha_t)))) * cos_prime_alpha

        return t

    def find_trail(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """
        Finds the pneumatic trail of the tyre for combined slip conditions.

        Parameters
        ----------
        SA : allowableData
            Slip angle.
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VC : allowableData, optional
            Contact patch speed (will default to ``LONGVL`` if not specified).
        VCX : allowableData, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        VS : allowableData, optional
            Slip speed magnitude (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        t : allowableData
            Pneumatic trail.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX
        PHI = 0.0 if PHI is None else PHI

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VC, VCX, VS = self._format_check([SA, SL, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            zeta_5 = self.turn_slip._find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_default

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC, VCX)

        # find coefficients
        BT, CT, DT, ET, alpha_t = self.__trail_main_routine(SA, FZ, P, IA, VCX, VS, zeta_5)

        # slip stiffness
        KYA = self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)
        KXK = self.gradient.find_slip_stiffness(FZ, P)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self.eps_K * np.sign(KYA)

        # corrected slip angle (4.E77) TODO: change to (A55)
        alpha_t_eq = np.sqrt(alpha_t ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_t)

        # pneumatic trail (4.E73)
        t = DT * self.cos(CT * self.atan(BT * alpha_t_eq - ET * (BT * alpha_t_eq - self.atan(BT * alpha_t_eq)))) * cos_prime_alpha
        return t

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __trail_main_routine(
            self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData,
            IA:  allowableData,
            VCX: allowableData,
            VS:  allowableData,
            zeta_5) -> list[allowableData]:
        """
        Function containing the main calculations for the pneumatic trail. To be used in ``find_trail`` and
        ``find_trail_pure``.
        """

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # scaled nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # degressive friction factor (4.E8)
        LMUY_star  = self.correction._find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # stiffness factor (4.E40) TODO QBZ6 changed to QBZ4
        BT = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) * (1.0 + self.QBZ5 * np.abs(gamma_star) + self.QBZ4 * gamma_star ** 2) * self.LYKA / LMUY_prime

        # TODO: Figure out which version for BT to use. QBZ4 is not is the book by Pacejka & Besselink, but it is in their 2010 paper
        # BT = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz ** 2) * (1.0 + QBZ4 + QBZ5 * np.abs(gamma_star) + QBZ6 * gamma_star ** 2) * LYKA / LMUY_prime

        # shape factor (4.E41)
        CT = self.QCZ1

        # static peak factor (4.E42) TODO
        DT0 = self.common._find_dt0(FZ, dfz, dpi, VCX, FZ0_prime, R0)

        # peak factor (4.E43)
        DT = DT0 * (1.0 + self.QDZ3 * np.abs(gamma_star) + self.QDZ4 * gamma_star ** 2) * zeta_5

        # horizontal shift (4.E35)
        S_HT = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma_star

        # corrected slip angle (4.E34)
        alpha_star = self.correction._find_alpha_star(SA, VCX)
        alpha_t = alpha_star + S_HT

        # curvature factor (4.E44)
        ET = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz ** 2) * (1.0 + (self.QEZ4 + self.QEZ5 * gamma_star) * np.pi / 2 * self.atan(BT * CT * alpha_t))

        return [BT, CT, DT, ET, alpha_t]
