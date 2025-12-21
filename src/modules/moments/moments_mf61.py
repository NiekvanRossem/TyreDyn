from src.utils.misc import allowableData
from typing import Literal
import numpy as np

class MomentsMF61:
    """
    Moments module for MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other modules
        self.turn_slip  = model.turn_slip
        self.friction   = model.friction
        self.gradient   = model.gradient
        self.forces     = model.forces
        self.trail      = model.trail

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    # ------------------------------------------------------------------------------------------------------------------#
    # PURE SLIP MOMENTS

    def find_mx_pure(
            self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the overturning couple for pure slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MX`` -- overturning couple.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VCX, VS = self._format_check([SA, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.forces.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # find overturning moment
        MX = self.__mx_main_routine(FY, FZ, P, IA)
        return MX

    def find_my_pure(
            self,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VX: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the rolling resistance couple for pure slip conditions.

        :param SL: slip ratio.
        :param FZ: vertical load.
        :param VX: longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MY`` -- rolling resistance couple.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P
        VX = self.LONGVL if VX is None else VX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SL, FZ, P, IA, VX = self._format_check([SL, FZ, P, IA, VX])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # calculate FX
        FX = self.forces.find_fx_pure(SL, FZ, P, IA, VS, angle_unit)

        # find rolling resistance moment
        MY = self.__my_main_routine(FX, FZ, P, IA, VX)
        return MY

    def find_mz_pure(
            self,
            SA:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the self-aligning couple for pure slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, will default to ``LONGVL`` if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param VS: slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MZ`` -- self-aligning couple.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_0 = 0.0  # (4.83)
            zeta_2 = self.turn_slip._find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6() # TODO
            zeta_7 = self.turn_slip._find_zeta_7()
            zeta_8 = self.turn_slip._find_zeta_8(FZ, P, IA, VS, angle_unit)
        else:
            zeta_0 = self.zeta_0_default
            zeta_2 = self.zeta_2_default
            zeta_4 = self.zeta_4_default
            zeta_6 = self.zeta_6_default
            zeta_7 = self.zeta_7_default
            zeta_8 = self.zeta_8_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VC, VCX, VS = self._format_check([SA, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # pneumatic trail TODO
        t = self.trail.find_trail_pure(SA, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # find side force
        FY = self.forces.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # cornering stiffness to self aligning couple (4.E48) TODO: move to gradient and figure out their purpose
        # KZAO = D_T0 * KYA

        # camber stiffness to self aligning couple (4.E49) TODO: move to gradient and figure out their purpose
        # KZCO = FZ * R0 * (self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi) * self.LKZC * LMUY_star - D_T0 * KYCO

        # residual self-aligning couple (4.E36)
        MZR = self.__mz_main_routine(SA, 0.0, FZ, P, IA, VC, VCX, VS, PHI, zeta_0, zeta_2, zeta_4, zeta_6, zeta_7,
                                     zeta_8, combined_slip=False, angle_unit=angle_unit)

        # self-aligning couple due to pneumatic trail (4.E32)
        MZ_prime = - t * FY

        # final self-aligning couple (4.E31)
        MZ = MZ_prime + MZR

        return MZ

    # ------------------------------------------------------------------------------------------------------------------#
    # COMBINED SLIP MOMENTS

    # TESTED
    def find_mx(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the overturning couple for combined slip conditions.

        :param PHI:
        :param SA: slip angle
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: ground speed (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MX`` -- overturning couple.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS = self._format_check([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.forces.find_fy(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # find overturning couple
        MX = self.__mx_main_routine(FY, FZ, P, IA)
        return MX

    def find_my(
            self,
            SA: allowableData,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VX: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the rolling resistance couple for combined slip conditions. Calculations according to Pacejka's MF 6.1.2
        model.

        :param SA: slip angle
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MY`` -- rolling resistance couple.
        """

        # assumed that difference between contact patch and wheel center speed is negligible as (eqn 7.4 from Pacejka)
        VCX = self.LONGVL if VX is None else VX

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P
        VX = self.LONGVL if VX is None else VX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VX = self._format_check([SA, SL, FZ, P, IA, VCX, VX])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # calculate FX
        FX = self.forces.find_fx(SA, SL, FZ, P, IA, VS, VCX, angle_unit)

        # find rolling resistance moment
        MY = self.__my_main_routine(FX, FZ, P, IA, VX)
        return MY

    def find_mz(
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
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the self-aligning couple for combined slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, will default to ``LONGVL`` if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param VS: slip speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``MZ`` -- self-aligning couple.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_2 = self.turn_slip._find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
            zeta_6 = self.turn_slip._find_zeta_6() # TODO
            zeta_7 = self.turn_slip._find_zeta_7()
            zeta_8 = self.turn_slip._find_zeta_8(FZ, P, IA, VS, angle_unit)
        else:
            zeta_2 = self.zeta_2_default
            zeta_4 = self.zeta_4_default
            zeta_6 = self.zeta_6_default
            zeta_7 = self.zeta_7_default
            zeta_8 = self.zeta_8_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VC, VCX, VS = self._format_check([SA, SL, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # scaled nominal loads
        FZ0_prime = FZ0 * self.LFZO

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)

        # corrected camber angle
        gamma_star = self.normalize._find_gamma_star(IA)

        # tyre forces
        FX = self.forces.find_fx(SA, SL, FZ, P, IA, VCX, VS, angle_unit)
        FY = self.forces.find_fy(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # side force with zero camber (4.E74)
        FY_prime = self.forces.find_fy(SA, SL, FZ, P, 0.0, VCX, VS, PHI, angle_unit)

        # pneumatic trail
        t = self.trail.find_trail(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # pneumatic scrub (4.E76)
        s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0_prime) + (self.SSZ3 + self.SSZ4 * dfz) * gamma_star) * self.LS

        # self-aligning couple from side force (4.E72)
        MZ_prime = -t * FY_prime

        # residual self-aligning couple
        MZR = self.__mz_main_routine(SA, SL, FZ, P, IA, VC, VCX, VS, PHI,
                                     zeta_0, zeta_2, zeta_4, zeta_6, zeta_7, zeta_8,
                                     combined_slip=True, angle_unit=angle_unit)

        # final self-aligning couple (4.E71)
        MZ = MZ_prime + MZR + s * FX
        return MZ

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __mx_main_routine(
            self,
            FY: allowableData,
            FZ: allowableData,
            P:  allowableData,
            IA: allowableData) -> allowableData:
        """Function containing the main ``MX`` calculation routine. To be used in ``find_mx`` and ``find_mx_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # normalize pressure
        dpi = self.normalize._find_dpi(P)

        # overturning couple (4.E69) -- FZ trig functions do not get corrected to degrees
        A = self.QSX1 * self.LVMX # TODO rename to be more clear
        B = self.QSX2 * IA * (1.0 + self.PPMX1 * dpi)
        C = self.QSX3 * FY / FZ0
        D = self.QSX4 * np.cos(self.QSX5 * np.atan2(self.QSX6 * FZ / FZ0, 1) ** 2)
        E = np.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * FY / FZ0, 1))
        F = self.QSX10 * np.atan2(self.QSX11 * FZ / FZ0, 1) * IA
        MX = R0 * FZ * (A - B + C + D * E + F) * self.LMX

        return MX

    def __my_main_routine(
            self,
            FX: allowableData,
            FZ: allowableData,
            P:  allowableData,
            IA: allowableData,
            VX: allowableData) -> allowableData:
        """Function containing the main ``MY`` calculation routine. To be used in ``find_my`` and ``find_my_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL
        FZ0 = self.FNOMIN
        P0  = self.NOMPRES

        # rolling resistance moment (4.E70)
        A = self.QSY1 # TODO rename to be more clear
        B = self.QSY2 * FX / FZ0
        C = self.QSY3 * np.abs(VX / V0)
        D = self.QSY4 * (VX / V0) ** 4
        E = (self.QSY5 + self.QSY6 * FZ / FZ0) * IA ** 2
        F = (FZ / FZ0) ** self.QSY7
        G = (P / P0) ** self.QSY8
        MY = FZ * R0 * (A + B + C + D + E) * F * G * self.LMY

        return MY

    def __mz_main_routine(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData,
            IA:  allowableData,
            VC:  allowableData,
            VCX: allowableData,
            VS:  allowableData,
            PHI: allowableData,
            zeta_0, zeta_2, zeta_4, zeta_6, zeta_7, zeta_8,
            combined_slip: bool = False,
            angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """Function containing the main ``MZ`` calculation routine. Used in ``find_mz`` and ``find_mz_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.normalize._find_dfz(FZ)
        dpi = self.normalize._find_dpi(P)

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # friction scaling factors
        LMUY_star  = self.correction._find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # cornering and camber stiffness
        KYA  = self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)
        KYCO = self.gradient.find_camber_stiffness(FZ, P)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self.eps_K * np.sign(KYA)

        # vertical shift for side force (4.E29)
        S_VY, S_VYg = self.common._find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.common._find_s_hy(dfz, KYA, KYCO, gamma_star, S_VYg, zeta_0, zeta_4)

        # horizontal shift for residual couple (4.E38)
        S_HF = S_HY + S_VY / KYA_prime

        # corrected slip angles (4.E3, 4.E37)
        alpha_star = self.correction._find_alpha_star(SA, VCX)
        alpha_r = alpha_star + S_HF

        # correction on the slip angle for combined slip
        if combined_slip:

            # slip stiffness
            KXK = self.gradient.find_slip_stiffness(FZ, P)

            # corrected slip angle (4.E78)
            alpha_r_eq = np.sqrt(alpha_r ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_r)
            alpha_used = alpha_r_eq
        else:
            alpha_used = alpha_r

        # friction scaling factor
        LMUY_star = self.correction._find_lmu_star(VS, V0, self.LMUY)

        # friction coefficient (4.E23)
        mu_y = self.friction.find_mu_y(FZ, LMUY_star, IA, VS, angle_unit)

        # peak factor (4.E22)
        D_Y = self.common._find_dy(mu_y, FZ, zeta_2)

        # cosine term correction factor
        cos_prime_alpha = self.correction._find_cos_prime_alpha(VC, VCX)

        # shape factor (4.E21)
        C_Y = self.common._find_cy()

        # stiffness factor (4.E26)
        B_Y = self.common._find_by(FZ, KYA, C_Y, D_Y)

        # stiffness factor for the residual couple (4.E45)
        B_R = (self.QBZ9 * self.LYKA / LMUY_star + self.QBZ10 * B_Y * C_Y) * zeta_6

        # shape factor for the residual couple (4.E46)
        C_R = zeta_7

        # peak factor for residual couple (4.E47)
        D_R = (FZ * R0 * ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES * zeta_2
                          + ((self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi)
                             + (self.QDZ10 + self.QDZ11 * dfz) * np.abs(gamma_star))
                          * gamma_star * self.LKZC * zeta_0) * LMUY_star
               * np.sign(VCX) * cos_prime_alpha + zeta_8 - 1.0)

        # residual self-aligning couple (4.E36)
        MZR = D_R * self.cos(C_R * self.atan(B_R * alpha_used)) * cos_prime_alpha

        return MZR
