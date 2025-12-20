import numpy as np
from typing import Union, Literal
import warnings

from src.models.base_tyre import TyreBase
from src.utils.misc import allowableData, check_format

# TODO: add turn slip correction [WIP]
# TODO: add low speed correction

class MF61(TyreBase):
    """
    Class definition for the Magic Formula 6.1 tyre model. Initialize an instance of this class by calling
    ``Tyre(<filename.tir>)``, where ``<filename.tir>`` is a TIR property file with ``FITTYP`` ``61`` or newer.

    This class contains functions to evaluate the tyre state based on a set of inputs.

    References:
      - Pacejka, H.B. & Besselink, I. (2012). *Tire and Vehicle Dynamics. Third Edition*. Elsevier.
        `doi: 10.1016/c2010-0-68548-8 <https://doi.org/10.1016/c2010-0-68548-8>`_
      - Marco Furlan (2025). *MFeval*. MATLAB Central File Exchange. Retrieved December 18, 2025.
        `mathworks.com/matlabcentral/fileexchange/63618-mfeval <https://mathworks.com/matlabcentral/fileexchange/63618-mfeval>`_
      - International Organization for Standardization (2011). *Road vehicles -- Vehicle dynamics and road-holding
        ability -- Vocabulary* (ISO standard No. 8855:2011)
        `iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en <https://www.iso.org/obp/ui/#iso:std:iso:8855:ed-2:v1:en>`_
    """

    def __init_from_data__(self, data, **settings):
        """
        Here some widely used standard values are stored, as well as terms that still need an implementation.
        """

        # run the initialization from the base class
        super().__init_from_data__(data, **settings)

        # turn slip correction factors if turn slip is disabled
        self.zeta_0_default = 1.0
        self.zeta_1_default = 1.0
        self.zeta_2_default = 1.0
        self.zeta_3_default = 1.0
        self.zeta_4_default = 1.0
        self.zeta_5_default = 1.0
        self.zeta_6_default = 1.0
        self.zeta_7_default = 1.0
        self.zeta_8_default = 1.0

        # correction factors to avoid singularities at low speed
        self.eps_x = 1e-6
        self.eps_K = 1e-6
        self.eps_V = 0.1  # set to 0.1 as suggested by Pacejka

        # scaling factor to control decaying friction with increasing speed (set to zero generally)
        self.LMUV = 0.0

        # low friction correction for friction coefficient scaling factor, set to 10 as suggested by Pacejka (4.E8)
        self.A_mu = 10.0

    #------------------------------------------------------------------------------------------------------------------#
    # FREE ROLLING FORCES

    # TESTED
    def find_fx_pure(
            self,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the longitudinal force for pure slip conditions.

        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``FX`` -- longitudinal force for pure slip.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SL, FZ, P, IA, VS = check_format([SL, FZ, P, IA, VS])

        # perform limit checks
        if self._check_limits:
            self._limit_check(None, SL, FZ, P, IA)

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # reference speed
        V0 = self.LONGVL

        # normalize inputs
        dfz = self.__find_dfz(FZ)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.__find_lmu_star(VS, V0, self.LMUX)

        # degressive friction factor (4.E8)
        LMUX_prime = self.__find_lmu_prime(LMUX_star)

        # horizontal shift (4.E17)
        S_HX = (self.PHX1 + self.PHX2 * dfz) * self.LHX

        # vertical shift (4.E18)
        S_VX = FZ * (self.PVX1 + self.PVX2 * dfz) * self.LVX * LMUX_prime * self.zeta_1

        # corrected slip ratio (4.E10)
        kappa_x = SL + S_HX

        # shape factor (4.E11)
        C_X = self.PCX1 * self.LCX

        # friction coefficient (4.E13)
        mu_x = self.find_mu_x(FZ, P, IA, VS, angle_unit)

        # peak factor (4.E12)
        D_X = mu_x * FZ * self.zeta_1

        # curvature factor (4.E14)
        E_X = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) * (1.0 - self.PEX4 * np.sign(kappa_x)) * self.LEX

        # slip stiffness (4.E15)
        KXK = self.find_slip_stiffness(FZ, P)

        # stiffness factor (4.E16)
        B_X = KXK / (C_X * D_X + self.eps_x)

        # Longitudinal force (4.E9) -- slip ratio trig functions do not get corrected to degrees
        FX = D_X * np.sin(C_X * self.atan(B_X * kappa_x - E_X * (B_X * kappa_x - np.atan2(B_X * kappa_x, 1)))) + S_VX

        return FX

    # TESTED
    def find_fy_pure(
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
        Finds the side force for pure slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return:
            ``FY`` -- side force.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_0 = 0.0 # (4.83)
            zeta_2 = self.__find_zeta_2(SA, FZ, PHI)
        else:
            zeta_0 = self.zeta_0_default
            zeta_2 = self.zeta_2_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VS, VCX = check_format([SA, FZ, P, IA, VS, VCX])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find normalized load and pressure
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # allows for large slip angles and reverse running (4.E3)
        alpha_star = self.__find_alpha_star(SA, VCX)

        # for spin due to camber angle (4.E4)
        gamma_star = self.__find_gamma_star(IA)
        
        # reference speed
        V0 = self.LONGVL

        # composite friction scaling factor (4.E7)
        LMUY_star = self.__find_lmu_star(VS, V0, self.LMUY)

        # degressive friction factor (4.E8)
        LMUY_prime = self.__find_lmu_prime(LMUY_star)

        # cornering stiffness (4.E25)
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # camber stiffness (4.E30)
        KYCO = self.find_camber_stiffness(FZ, P)

        # vertical shifts (4.E29)
        S_VY, S_VYg = self.__find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2) # TODO: maybe remove dfz from this function

        # horizontal shift (4.E27)
        S_HY = self.__find_s_hy(dfz, KYA, KYCO, gamma_star, S_VYg, zeta_0, zeta_4)

        # corrected slip angle (4.E20)
        alpha_y = alpha_star + S_HY

        # shape factor (4.E21)
        C_Y = self.__find_cy()

        # friction coefficient (4.E23)
        mu_y = self.find_mu_y(FZ, P, IA, VS, angle_unit)

        # peak factor (4.E22)
        D_Y = self.__find_dy(mu_y, FZ, zeta_2)

        # curvature factor (4.E24)
        E_Y = (self.PEY1 + self.PEY2 * dfz) * (1.0 + self.PEY5 * gamma_star ** 2 - (self.PEY3 + self.PEY4 * gamma_star) * np.sign(alpha_y)) * self.LEY

        # stiffness factor (4.E26)
        B_Y = self.__find_by(FZ, KYA, C_Y, D_Y)

        # lateral force (4.E19)
        FY = D_Y * self.sin(C_Y * self.atan(B_Y * alpha_y - E_Y * (B_Y * alpha_y - self.atan(B_Y * alpha_y)))) + S_VY

        return FY

    #------------------------------------------------------------------------------------------------------------------#
    # COMBINED SLIP FORCES

    # TESTED
    def find_fx(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Finds the longitudinal force for combined slip conditions.

        :param SL: slip ratio.
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``FX`` -- longitudinal force.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS = check_format([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # normalized vertical load
        dfz = self.__find_dfz(FZ)

        # horizontal shift (4.E57)
        S_HXA = self.RHX1

        # corrected slip angles (4.E53)
        alpha_star = self.__find_alpha_star(SA, VCX)
        alpha_s = alpha_star + S_HXA

        # corrected camber angle (4.E4)
        gamma_star = self.__find_gamma_star(IA)

        # stiffness factor (4.E54) -- slip ratio trig functions do not get corrected to degrees
        B_XA = (self.RBX1 + self.RBX3 * gamma_star ** 2) * np.cos(np.atan2(self.RBX2 * SL, 1)) * self.LXAL

        # shape factor (4.E55)
        C_XA = self.RCX1

        # curvature factor (4.E56)
        E_XA = self.REX1 + self.REX2 * dfz

        # static correction (4.E52)
        GXAO = np.cos(C_XA * self.atan(B_XA * S_HXA - E_XA * (B_XA * S_HXA - self.atan(B_XA * S_HXA))))

        # force correction factor (4.E51)
        GXA = np.cos(C_XA * self.atan(B_XA * alpha_s - E_XA * (B_XA * alpha_s - self.atan(B_XA * alpha_s)))) / GXAO

        # force for pure slip
        FX0 = self.find_fx_pure(SL, FZ, P, IA, VS, angle_unit)

        # longitudinal force for combined slip (4.E50)
        FX = FX0 * GXA
        return FX

    # TESTED
    def find_fy(
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
        Finds the side force for combined slip conditions.

        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param PHI: turn slip (optional).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``FY`` -- side force.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_2 = self.__find_zeta_2(SA, FZ, PHI)
        else:
            zeta_2 = self.zeta_2_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS = check_format([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # normalized vertical load
        dfz = self.__find_dfz(FZ)

        # corrected slip angle (4.E53)
        alpha_star = self.__find_alpha_star(SA, VCX)

        # corrected camber angle (4.E4)
        gamma_star = self.__find_gamma_star(IA)

        # side force for pure slip
        FY0 = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # lateral friction coefficient
        mu_y = self.find_mu_y(FZ, P, IA, VS, angle_unit)

        # stiffness factor (4.E62)
        B_YK = (self.RBY1 + self.RBY4 * gamma_star ** 2) * self.cos(self.atan(self.RBY2 * (alpha_star - self.RBY3))) * self.LYKA

        # shape factor (4.E63)
        C_YK = self.RCY1

        # peak factor (4.E67)
        D_VYK = mu_y * FZ * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma_star) * self.cos(self.atan(self.RVY4 * alpha_star)) * zeta_2

        # curvature factor (4.E64)
        E_YK = self.REY1 + self.REY2 * dfz

        # horizontal shift (4.E65)
        S_HYK = self.RHY1 + self.RHY2 * dfz

        # vertical shift (4.E66) -- slip ratio trig functions do not get corrected to degrees
        S_VYK = D_VYK * np.sin(self.RVY5 * np.atan2(self.RVY6 * SL, 1)) * self.LVYKA

        # corrected slip ratio (4.E61)
        kappa_s = SL + S_HYK

        # static correction (4.E60) -- slip ratio trig functions do not get corrected to degrees
        GYKO = np.cos(C_YK * np.atan2(B_YK * S_HYK - E_YK * (B_YK * S_HYK - np.atan2(B_YK * S_HYK, 1)), 1))

        # force correction (4.E59) -- slip ratio trig functions do not get corrected to degrees
        GYK = np.cos(C_YK * np.atan2(B_YK * kappa_s - E_YK * (B_YK * kappa_s - np.atan2(B_YK * kappa_s, 1)), 1)) / GYKO

        # side force for combined slip (4.E58)
        FY = FY0 * GYK + S_VYK
        return FY

    #------------------------------------------------------------------------------------------------------------------#
    # FREE ROLLING MOMENTS

    # TESTED
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
            SA, FZ, P, IA, VCX, VS = check_format([SA, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

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
            SL, FZ, P, IA, VX = check_format([SL, FZ, P, IA, VX])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # calculate FX
        FX = self.find_fx_pure(SL, FZ, P, IA, VS, angle_unit)

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
            zeta_2 = self.__find_zeta_2(SA, FZ, PHI)
        else:
            zeta_2 = self.zeta_2_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VC, VCX, VS = check_format([SA, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # pneumatic trail
        t = self.find_trail_pure(SA, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # find side force
        FY = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # cornering stiffness to self aligning couple (4.E48)
        #KZAO = D_T0 * KYA

        # camber stiffness to self aligning couple (4.E49)
        #KZCO = FZ * R0 * (self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi) * self.LKZC * LMUY_star - D_T0 * KYCO

        # residual self-aligning couple (4.E36)
        MZR = self.__mz_main_routine(SA, 0.0, FZ, P, IA, VC, VCX, VS, PHI, zeta_0, zeta_2, zeta_4, combined_slip=False,
                                     angle_unit=angle_unit)

        # self-aligning couple due to pneumatic trail (4.E32)
        MZ_prime = - t * FY

        # final self-aligning couple (4.E31)
        MZ = MZ_prime + MZR

        return MZ

    #------------------------------------------------------------------------------------------------------------------#
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
            SA, SL, FZ, P, IA, VCX, VS = check_format([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # find side force
        FY = self.find_fy(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # find overturning couple
        MX = self.__mx_main_routine(FY, FZ, P, IA)
        return MX

    def find_my(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VX:  allowableData = None,
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
            SA, SL, FZ, P, IA, VCX, VX = check_format([SA, SL, FZ, P, IA, VCX, VX])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # calculate FX
        FX = self.find_fx(SA, SL, FZ, P, IA, VS, VCX, angle_unit)

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
            zeta_2 = self.__find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.__find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
        else:
            zeta_2 = self.zeta_2_default
            zeta_4 = self.zeta_4_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VC, VCX, VS = check_format([SA, SL, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # scaled nominal loads
        FZ0_prime = FZ0 * self.LFZO

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # tyre forces
        FX = self.find_fx(SA, SL, FZ, P, IA, VCX, VS, angle_unit)
        FY = self.find_fy(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # side force with zero camber (4.E74)
        FY_prime = self.find_fy(SA, SL, FZ, P, 0.0, VCX, VS, PHI, angle_unit)

        # pneumatic trail
        t = self.find_trail(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # pneumatic scrub (4.E76)
        s = R0 * (self.SSZ1 + self.SSZ2 * (FY / FZ0_prime) + (self.SSZ3 + self.SSZ4 * dfz) * gamma_star) * self.LS

        # self-aligning couple from side force (4.E72)
        MZ_prime = -t * FY_prime

        # residual self-aligning couple
        MZR = self.__mz_main_routine(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, zeta_0, zeta_2, zeta_4, combined_slip=True,
                                     angle_unit=angle_unit)

        # final self-aligning couple (4.E71)
        MZ = MZ_prime + MZR + s * FX
        return MZ

    #------------------------------------------------------------------------------------------------------------------#
    # COMBINED FUNCTIONS

    # TESTED
    def find_forces(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the force vector for combined slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``F`` -- list containing the force vector components. Order is ``X``, ``Y``, ``Z``.
        """

        # find planar forces
        FX = self.find_fx(SA, SL, FZ, P, IA, VCX, VS, angle_unit=angle_unit)
        FY = self.find_fy(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit=angle_unit)
        return [FX, FY, FZ]

    def find_moments(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the moment vector for combined slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param VX: longitudinal speed.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: list of tyre moments. Order is ``X``, ``Y``, ``Z``.
        """

        MX = self.find_mx(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MY = self.find_my(SA, SL, FZ, P, IA, VX, angle_unit)
        MZ = self.find_mz(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)
        return [MX, MY, MZ]

    def find_force_moment(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> list[allowableData]:
        """
        Finds the total force and moment vector of the tyre for combined slip conditions.

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param VX: longitudinal speed.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param VS: slip speed magnitude (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: list of tyre forces and moments. Order is ``FX``, ``FY``, ``FZ``, ``MX``, ``MY``, ``MZ``.
        """

        [FX, FY, FZ] = self.find_forces(SA, SL, FZ, P, IA, VCX, VS, PHI, angle_unit)
        [MX, MY, MZ] = self.find_moments(SA, SL, FZ, VX, P, IA, VCX, VS, PHI, angle_unit)
        return [FX, FY, FZ, MX, MY, MZ]

    def find_lateral_output(
            self,
            SA:  allowableData,
            FZ:  allowableData,
            N:   allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the free rolling outputs commonly used in lateral vehicle tyre_models.

        :param PHI:
        :param N:
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, will default to ``LONGVL`` if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param VS:  slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: Output state. Order is ``FY``, ``MX``, ``MZ``, ``RL``, ``sigma_y``.
        """

        FY = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MX = self.find_mx_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)
        MZ = self.find_mz_pure(SA, FZ, P, IA, VC, VCX, VS, PHI)
        RL = self.find_loaded_radius(0.0, FY, FZ, N, P)
        sigma_y = self.find_lateral_relaxation(FZ, P, IA, PHI, angle_unit)
        return [FY, MX, MZ, RL, sigma_y]

    def find_longitudinal_output(
            self,
            SL: allowableData,
            FZ: allowableData,
            P:  allowableData = None,
            IA: allowableData = 0.0,
            VS: allowableData = 0.0,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the pure slip forces and moments commonly used in longitudinal vehicle tyre_models.

        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VS: slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: Longitudinal tyre forces and moments. Order is ``FX``, ``MY``.
        """

        FX = self.find_fx_pure(SL, FZ, P, IA, VS, angle_unit)
        MY = self.find_my_pure(SL, FZ, P, IA, VS, angle_unit)
        return [FX, MY]

    def find_full_state(
            self,
            SA:  allowableData,
            SL:  allowableData,
            FZ:  allowableData,
            VX:  allowableData,
            N:   allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VC:  allowableData = None,
            VCX: allowableData = None,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["rad", "deg"] = "rad") -> list[allowableData]:
        """
        Finds the full output state of the tyre. Not recommended to use this in performance-sensitive vehicle models, as
        some functions are called multiple times.

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param VX: longitudinal speed.
        :param N: wheel angular speed.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, will default to ``LONGVL`` if not specified).
        :param VCX: contact patch longitudinal speed (optional, will default to ``LONGVL`` if not specified).
        :param VS:  slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: list containing the full state of the tyre. See documentation for the order.
        """

        # turn slip correction
        if self._use_turn_slip is True and PHI is not None:
            zeta_2 = self.__find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.__find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
        else:
            zeta_2 = self.zeta_2_default
            zeta_4 = self.zeta_4_default

        # force and moment vector
        [FX, FY, FZ, MX, MY, MZ] = self.find_force_moment(SA, SL, FZ, VX, P, IA, VCX, VS, PHI, angle_unit)

        # residual self-aligning couple
        MZR = self.__mz_main_routine(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, zeta_0, zeta_2, zeta_4, combined_slip=True, angle_unit=angle_unit)

        # free, loaded, and effective radii, and deflection
        R_omega = self.find_free_radius(N) # NOT USED IN COMPATIBILITY MODE
        RE      = self.find_effective_radius(FZ, N, P)
        RL      = self.find_loaded_radius(FX, FY, FZ, N, P)
        rho     = self.find_deflection(FX, FY, FZ, N, P)

        # pneumatic trail
        t = self.find_trail(SA, SL, FZ, P, IA, VC, VCX, VS, PHI, angle_unit)

        # friction coefficients
        mu_x = self.find_mu_x(FZ, P, IA, VS, angle_unit)
        mu_y = self.find_mu_y(FZ, P, IA, VS, angle_unit)

        # contact patch dimensions
        a, b = self.find_contact_patch(FZ)

        # tyre stiffness
        Cx = self.find_longitudinal_stiffness(FZ, P)
        Cy = self.find_lateral_stiffness(FZ, P)
        Cz = self.find_vertical_stiffness(P)

        # slip stiffness
        KXK = self.find_slip_stiffness(FZ, P)
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # relaxation length
        sigma_x = self.find_longitudinal_relaxation(FZ, P)
        sigma_y = self.find_lateral_relaxation(FZ, P, IA, PHI, angle_unit)

        # instantaneous slip stiffness
        iKYA = self.find_instant_kya(SA, FY)
        iKXK = self.find_instant_kxk(SL, FX)

        # assemble final output
        if self.use_mfeval_mode:

            # compatibility mode. Output vector has the same order as MFeval
            output = [FX, FY, FZ, MX, MY, MZ, SL, SA, IA, PHI, VX, P, RE, rho, 2*a,
                t, mu_x, mu_y, N, RL, 2*b, MZR, Cx, Cy, Cz, KYA, sigma_x, sigma_y, iKYA, KXK]
        else:

            # more organized output vector
            output = [
                FX, FY, FZ,                 # FORCES
                MX, MY, MZ,                 # MOMENTS
                SL, SA, IA, PHI, VX, P, N,  # INPUT STATE
                R_omega, RE, rho, RL,       # RADII
                2*a, 2*b,                   # CONTACT PATCH
                t,                          # TRAIL
                mu_x, mu_y,                 # FRICTION COEFFICIENT
                MZR,                        # RESIDUAL MOMENT
                Cx, Cy, Cz,                 # TYRE STIFFNESS
                KYA, iKYA, KXK, iKXK,       # SLIP STIFFNESS
                sigma_x, sigma_y            # RELAXATION LENGTHS
                ]
        return output

    #------------------------------------------------------------------------------------------------------------------#
    # TYRE RADII

    def find_deflection(
            self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Finds the vertical deflection of the tyre. A positive value signifies compression.

        :param FX: longitudinal force.
        :param FY: lateral force.
        :param FZ: vertical load.
        :param N: wheel angular speed.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``rho`` -- vertical deflection of the tyre. Compression is positive.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FX, FY, FZ, N, P = check_format([FX, FY, FZ, N, P])

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL
        FZ0 = self.FNOMIN
        CZ0 = self.VERTICAL_STIFFNESS

        # find QFZ1 from CZ0 (A3.4)
        Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)

        # normalize tyre pressure
        dpi = self.__find_dpi(P)

        # inputs affecting the radius (A3.3) TODO: equation 4.E68 adds extra camber terms to it.
        speed_effect = self.Q_V2 * np.abs(N) * R0 / V0
        fx_effect = (self.Q_FCX * FX / FZ0) ** 2
        fy_effect = (self.Q_FCY * FY / FZ0) ** 2
        pressure_effect = (1.0 + self.PFZ1 * dpi) * FZ0

        # solve via the ABC formula
        A = - self.Q_FZ2 / (R0 ** 2)
        B = - Q_FZ1 / R0
        C = FZ / ((1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect)
        rho = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        # display warning if only imaginary solutions can be found for a datapoint
        check_root = B ** 2 - 4 * A * C
        if not isinstance(check_root, np.ndarray):
            if isinstance(check_root, list):
                check_root = np.array(check_root)
            else:
                check_root = np.array([check_root])
        if any(check_root < 0.0):
            warnings.warn("No real solution found for the tyre deflection!")

        # apply proper limits to avoid dividing by zero
        rho = np.clip(rho, 1e-6, np.inf)

        # TODO: add bottoming out check

        return rho

    def find_effective_radius(
            self,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Finds the effective tyre radius, to be used for calculating the slip ratio.

        :param FZ: vertical load.
        :param N: angular speed of the wheel.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``RE`` -- effective tyre radius.
        """

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, N, P = check_format([FZ, N, P])

        # unpack tyre properties
        FZ0 = self.FNOMIN

        # vertical stiffness
        CZ = self.find_vertical_stiffness(P)

        # loaded radius
        R_omega = self.find_free_radius(N)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = R_omega - FZ0 / CZ * (self.FREFF * FZ / FZ0 + self.DREFF * np.atan2(self.BREFF * FZ / FZ0, 1))
        return RE

    def find_free_radius(self, N: allowableData) -> allowableData:
        """
        Finds the free rolling radius, which capture the tyre growth as it spins up. Calculations according to Pacejka's
        MF 6.1.2.

        :param N: angular speed of the wheel.
        :return: ``R_omega`` -- free rolling radius.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            N = check_format(N)

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # free rolling radius (A3.1)
        R_omega = R0 * (self.Q_RE0 + self.Q_V1 * (R0 * N / V0) ** 2)
        return R_omega

    def find_loaded_radius(
            self,
            FX: allowableData,
            FY: allowableData,
            FZ: allowableData,
            N:  allowableData,
            P:  allowableData = None) -> allowableData:
        """
        Finds the loaded radius of the tyre. Calculations according to Pacejka's MF 6.1.2.

        :param FX: longitudinal force.
        :param FY: lateral force.
        :param FZ: vertical load.
        :param N: wheel angular speed.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``RL`` -- loaded rolling radius.
        """

        # set default value for optional argument
        if self._check_format:
            P = self.INFLPRES if P is None else P

        # free radius
        R_omega = self.find_free_radius(N)

        # deflection
        rho = self.find_deflection(FX, FY, FZ, N, P)

        # loaded radius
        RL = R_omega - rho

        return RL

    #------------------------------------------------------------------------------------------------------------------#
    # FRICTION COEFFICIENTS

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
            FZ, P, IA, VS = check_format([FZ, P, IA, VS])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # unpack tyre properties
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.__find_lmu_star(VS, V0, self.LMUX)

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
            FZ, P, IA, VS = check_format([FZ, P, IA, VS])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # unpack tyre properties
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # composite friction scaling factor (4.E7)
        LMUY_star = self.__find_lmu_star(VS, V0, self.LMUY)

        # lateral friction coefficient (4.E23)
        mu_y = ((self.PDY1 + self.PDY2 * dfz) * (1.0 + self.PPY3 * dpi + self.PPY4 * dpi ** 2)
                * (1.0 - self.PDY3 * gamma_star ** 2) * LMUY_star)

        return mu_y

    #------------------------------------------------------------------------------------------------------------------#
    # TYRE STIFFNESS

    def find_lateral_stiffness(self, FZ: allowableData, P: allowableData = None) -> allowableData:
        """
        Finds the lateral stiffness of the tyre, adjusted for load and pressure.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``Cy`` -- lateral stiffness.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = check_format([FZ, P])

        # unpack tyre properties
        Cy0 = self.LATERAL_STIFFNESS

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # lateral stiffness (A3.10)
        Cy = Cy0 * (1.0 + self.PCFY1 * dfz + self.PCFY2 * dfz ** 2) * (1.0 + self.PCFY3 * dpi)
        return Cy

    def find_longitudinal_stiffness(self, FZ: allowableData, P: allowableData = None) -> allowableData:
        """
        Finds the longitudinal stiffness of the tyre, adjusted for load and pressure.

        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).

        :return: ``Cx`` -- longitudinal stiffness.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = check_format([FZ, P])

        # unpack tyre properties
        Cx0 = self.LONGITUDINAL_STIFFNESS

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # lateral stiffness (A3.10)
        Cx = Cx0 * (1.0 + self.PCFX1 * dfz + self.PCFX2 * dfz ** 2) * (1.0 + self.PCFX3 * dpi)
        return Cx

    def find_vertical_stiffness(self, P: allowableData) -> allowableData:
        """
        Finds the vertical tyre stiffness, adjusted for pressure.

        :param P: tyre pressure.

        :return: ``CZ`` -- vertical tyre stiffness.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            P = check_format(P)

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS

        # normalize pressure
        dpi = self.__find_dpi(P)

        # current vertical rate (A3.5)
        CZ = CZ0 * (1.0 + self.PFZ1 * dpi)

        return CZ

    #------------------------------------------------------------------------------------------------------------------#
    # CONTACT PATCH DIMENSIONS

    def find_contact_patch(self, FZ: allowableData) -> list[allowableData]:
        """
        Finds the contact patch dimensions.

        :param FZ: vertical load.

        :return: ``a``, ``b`` -- ellipse radii of the contact patch.
        """

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ = check_format(FZ)

        # unpack tyre parameters
        R0 = self.UNLOADED_RADIUS
        W  = self.WIDTH

        # vertical stiffness
        CZ = self.find_vertical_stiffness(P)

        # length (A3.7)
        a = R0 * (self.Q_RA2 * FZ / (CZ * R0) + self.Q_RA1 * np.sqrt(FZ / (CZ * R0)))

        # half width (A3.8)
        b = W * (self.Q_RB2 * FZ / (CZ * R0) + self.Q_RB1 * (FZ / (CZ * R0)) ** (1/3))

        return [a, b]

    #------------------------------------------------------------------------------------------------------------------#
    # PNEUMATIC TRAIL

    # TESTED
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
        Finds the pneumatic trail of the tyre.

        :param PHI:
        :param SA: slip angle.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, if not selected the ``LONGVL`` parameter is used).
        :param VCX: contact patch longitudinal speed (optional, if not selected the ``LONGVL`` parameter is used).
        :param VS: slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``t`` -- pneumatic trail.
        """

        if self._use_turn_slip is True and PHI is not None:
            zeta_5 = self.__find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_5_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VC, VCX, VS = check_format([SA, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # cosine term correction factor
        cos_prime_alpha = self.__find_cos_prime_alpha(VC, VCX)

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

        :param PHI:
        :param SA: slip angle.
        :param SL: slip ratio.
        :param FZ: vertical load.
        :param P: tyre pressure (optional, if not selected the ``INFLPRES`` parameter is used).
        :param IA: camber angle with respect to the ground plane (optional, will default to zero if not specified).
        :param VC: contact patch speed (optional, will default to ``LONGVL`` if not specified).
        :param VCX: contact patch longitudinal speed (optional, if not selected the ``LONGVL`` parameter is used).
        :param VS: slip speed (optional, will default to zero if not specified).
        :param angle_unit: unit of the angles (optional, set to ``"deg"`` if your input arrays are specified in degrees).

        :return: ``t`` -- pneumatic trail.
        """

        if self._use_turn_slip is True and PHI is not None:
            zeta_5 = self.__find_zeta_5(PHI)
        else:
            zeta_5 = self.zeta_5_default

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VC  = self.LONGVL if VC is None else VC
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VC, VCX, VS = check_format([SA, SL, FZ, P, IA, VC, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # cosine term correction factor
        cos_prime_alpha = self.__find_cos_prime_alpha(VC, VCX)

        # find coefficients
        BT, CT, DT, ET, alpha_t = self.__trail_main_routine(SA, FZ, P, IA, VCX, VS, zeta_5)

        # slip stiffness
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)
        KXK = self.find_slip_stiffness(FZ, P)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self.eps_K * np.sign(KYA)

        # corrected slip angle (4.E77) TODO: change to (A55)
        alpha_t_eq = np.sqrt(alpha_t ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_t)

        # pneumatic trail (4.E73)
        t = DT * self.cos(CT * self.atan(BT * alpha_t_eq - ET * (BT * alpha_t_eq - self.atan(BT * alpha_t_eq)))) * cos_prime_alpha
        return t

    #------------------------------------------------------------------------------------------------------------------#
    # SLIP AND CAMBER STIFFNESS

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
            zeta_3 = self.__find_zeta_3(PHI)
        else:
            zeta_3 = self.zeta_3_default

        # set default values for optional arguments
        P  = self.INFLPRES if P is None else P

        # unpack tyre properties
        FZ0 = self.FNOMIN

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P, IA = check_format([FZ, P, IA])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # normalize pressure
        dpi = self.__find_dpi(P)

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
        P  = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            FZ, P = check_format([FZ, P])

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

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
            FZ, P = check_format([FZ, P])

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

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

    #------------------------------------------------------------------------------------------------------------------#
    # RELAXATION LENGTHS

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
            FZ, P, IA = check_format([FZ, P, IA])

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # cornering stiffness
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # lateral stiffness
        Cy = self.find_lateral_stiffness(FZ, P)

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
            FZ, P = check_format([FZ, P])

        # slip stiffness
        KXK = self.find_slip_stiffness(FZ, P)

        # longitudinal stiffness
        Cx = self.find_longitudinal_stiffness(FZ, P)

        # longitudinal relaxation length (A3.9)
        sigma_x = KXK / Cx
        return sigma_x

    #------------------------------------------------------------------------------------------------------------------#
    # INTERNAL FUNCTIONS

    def __find_alpha_star(self, SA: allowableData, VCX: allowableData) -> allowableData:
        """Finds the corrected slip angle."""

        # allows for large slip angles and reverse running (4.E3)
        if self._use_alpha_star:
            alpha_star = self.tan(SA) * np.sign(VCX)
        else:
            alpha_star = SA
        return alpha_star

    def __find_cos_prime_alpha(self, VC: allowableData, VCX: allowableData) -> allowableData:
        """Finds the correction factor for cosine terms when dealing with large slip angles."""

        # corrected wheel center speed (4.E6a)
        VC_prime = VC + self.eps_V

        # cosine correction (4.E6)
        cos_prime_alpha = VCX / VC_prime
        return cos_prime_alpha

    def __find_by(self, FZ: allowableData, KYA: allowableData, CY: allowableData, DY: allowableData) -> allowableData:
        """Finds the stiffness factor for the side force."""

        # side force stiffness factor (4.E26)
        eps_y = self.__find_eps_y(FZ)
        BY = KYA / (CY * DY + eps_y)
        return BY

    def __find_cy(self) -> allowableData:
        """Finds the shape factor for the side force."""

        # (4.E21)
        CY = self.PCY1 * self.LCY
        return CY

    def __find_dfz(self, FZ: allowableData) -> allowableData:
        """Finds the normalized vertical load."""

        # unpack parameters
        LFZ0 = self.LFZO
        FZ0 = self.FNOMIN

        # scale nominal load (4.E1)
        FZ0_2 = LFZ0 * FZ0

        # find normalized vertical load (4.E2a)
        dfz = (FZ - FZ0_2) / FZ0_2

        return dfz

    def __find_dpi(self, P: allowableData) -> allowableData:
        """Finds the normalized tyre pressure."""

        # extract parameters
        P0 = self.NOMPRES

        # normalized pressure (4.E2b)
        dpi = (P - P0) / P0
        return dpi

    def __find_dt0(self, FZ: allowableData, dfz: allowableData, dpi: allowableData, VCX: allowableData, FZ0_prime: allowableData, R0: Union[int, float]) -> allowableData:
        """Finds the static peak factor."""

        # (4.E42) TODO
        DT0 = FZ * (R0 / FZ0_prime) * (self.QDZ1 + self.QDZ2 * dfz) * (1.0 - self.PPZ1 * dpi) * self.LTR * np.sign(VCX)
        return DT0

    @staticmethod
    def __find_dy(mu_y: allowableData, FZ: allowableData, zeta_2) -> allowableData:
        """Finds the peak factor for the side force."""

        # (4.E22)
        DY = mu_y * FZ * zeta_2
        return DY

    def __find_eps_y(self, FZ):
        """Difference between camber and turn slip response."""

        if self._use_turn_slip:

            # normalize load
            dfz = self.__find_dfz(FZ)

            # difference between camber and turn slip response (4.90)
            eps_y = self.PECP1 * (1.0 + self.PECP2 * dfz)

        else:
            eps_y = 1e-6

        return eps_y

    def __find_gamma_star(self, IA: allowableData) -> allowableData:
        """Finds the corrected inclination angle."""

        # (4.E4)
        if self._use_gamma_star:
            gamma_star = self.sin(IA)
        else:
            gamma_star = IA
        return gamma_star

    def __find_lmu_prime(self, LMU_star: allowableData) -> allowableData:
        """Finds the composite friction scaling factor."""

        lmu_prime = self.A_mu * LMU_star / (1.0 + (self.A_mu - 1.0) * LMU_star)
        return lmu_prime

    def __find_lmu_star(self, VS: allowableData, V0: float, LMU: float) -> allowableData:
        """Finds the composite friction scaling factor."""

        # (4.E7)
        if self._use_lmu_star:
            LMU_star = LMU / (1.0 + self.LMUV * VS / V0)
        else:
            LMU_star = LMU
        return LMU_star

    def __find_s_hy(self, dfz: allowableData, KYA: allowableData, KYCO: allowableData, gamma_star: allowableData, S_VYg: allowableData, zeta_0, zeta_4) -> allowableData:
        """Finds the horizontal shift for the side force."""

        # (4.E27)
        S_HY = ((self.PHY1 + self.PHY2 * dfz) * self.LHY + (KYCO * gamma_star - S_VYg)
                / (KYA + self.eps_K) * zeta_0 + zeta_4 - 1.0)
        return S_HY

    def __find_s_vy(self, FZ: allowableData, dfz: allowableData, gamma_star: allowableData, LMUY_prime: allowableData, zeta_2) -> allowableData:
        """Finds the vertical shifts for the side force."""

        # vertical shift due to camber (4.E28)
        S_VYg = FZ * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta_2

        # total vertical shift (4.E29)
        S_VY = FZ * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta_2 + S_VYg
        return S_VYg, S_VY

    def __mx_main_routine(self, FY: allowableData, FZ: allowableData, P: allowableData, IA: allowableData) -> allowableData:
        """Function containing the main ``MX`` calculation routine. To be used in ``find_mx`` and ``find_mx_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN

        # normalize pressure
        dpi = self.__find_dpi(P)

        # overturning couple (4.E69) -- FZ trig functions do not get corrected to degrees
        A = self.QSX1 * self.LVMX
        B = self.QSX2 * IA * (1.0 + self.PPMX1 * dpi)
        C = self.QSX3 * FY / FZ0
        D = self.QSX4 * np.cos(self.QSX5 * np.atan2(self.QSX6 * FZ / FZ0, 1) ** 2)
        E = np.sin(self.QSX7 * IA + self.QSX8 * np.atan2(self.QSX9 * FY / FZ0, 1))
        F = self.QSX10 * np.atan2(self.QSX11 * FZ / FZ0, 1) * IA
        MX = R0 * FZ * (A - B + C + D * E + F) * self.LMX

        return MX

    def __my_main_routine(self, FX: allowableData, FZ: allowableData, P: allowableData, IA: allowableData, VX: allowableData) -> allowableData:
        """Function containing the main ``MY`` calculation routine. To be used in ``find_my`` and ``find_my_pure``."""

        # unpack tyre properties
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL
        FZ0 = self.FNOMIN
        P0  = self.NOMPRES

        # rolling resistance moment (4.E70)
        A = self.QSY1
        B = self.QSY2 * FX / FZ0
        C = self.QSY3 * np.abs(VX / V0)
        D = self.QSY4 * (VX / V0) ** 4
        E = (self.QSY5 + self.QSY6 * FZ / FZ0) * IA ** 2
        F = (FZ / FZ0) ** self.QSY7
        G = (P / P0) ** self.QSY8
        MY = FZ * R0 * (A + B + C + D + E) * F * G * self.LMY

        return MY

    def __mz_main_routine(self, SA: allowableData, SL: allowableData, FZ: allowableData, P: allowableData, IA: allowableData, VC: allowableData, VCX: allowableData, VS: allowableData, PHI, zeta_0, zeta_2, zeta_4, combined_slip: bool = False, angle_unit: Literal["rad", "deg"] = "rad") -> allowableData:
        """Function containing the main ``MZ`` calculation routine. Used in ``find_mz`` and ``find_mz_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # friction scaling factors
        LMUY_star  = self.__find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.__find_lmu_prime(LMUY_star)

        # cornering and camber stiffness
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)
        KYCO = self.find_camber_stiffness(FZ, P)

        # corrected cornering stiffness (4.E39)
        KYA_prime = KYA + self.eps_K * np.sign(KYA)

        # vertical shift for side force (4.E29)
        S_VY, S_VYg = self.__find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.__find_s_hy(dfz, KYA, KYCO, gamma_star, S_VYg, zeta_0, zeta_4)

        # horizontal shift for residual couple (4.E38)
        S_HF = S_HY + S_VY / KYA_prime

        # corrected slip angles (4.E3, 4.E37)
        alpha_star = self.__find_alpha_star(SA, VCX)
        alpha_r = alpha_star + S_HF

        # correction on the slip angle for combined slip
        if combined_slip:

            # slip stiffness
            KXK = self.find_slip_stiffness(FZ, P)

            # corrected slip angle (4.E78)
            alpha_r_eq = np.sqrt(alpha_r ** 2 + (KXK / KYA_prime) ** 2 * SL ** 2) * np.sign(alpha_r)
            alpha_used = alpha_r_eq
        else:
            alpha_used = alpha_r

        # friction scaling factor
        LMUY_star = self.__find_lmu_star(VS, V0, self.LMUY)

        # friction coefficient (4.E23)
        mu_y = self.find_mu_y(FZ, LMUY_star, IA, VS, angle_unit)

        # peak factor (4.E22)
        D_Y = self.__find_dy(mu_y, FZ, zeta_2)

        # cosine term correction factor
        cos_prime_alpha = self.__find_cos_prime_alpha(VC, VCX)

        # shape factor (4.E21)
        C_Y = self.__find_cy()

        # stiffness factor (4.E26)
        B_Y = self.__find_by(FZ, KYA, C_Y, D_Y)

        # stiffness factor for the residual couple (4.E45)
        B_R = (self.QBZ9 * self.LYKA / LMUY_star + self.QBZ10 * B_Y * C_Y) * self.zeta_6

        # shape factor for the residual couple (4.E46)
        C_R = self.zeta_7

        # peak factor for residual couple (4.E47)
        D_R = (FZ * R0 * ((self.QDZ6 + self.QDZ7 * dfz) * self.LRES * zeta_2
                          + ((self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi)
                             + (self.QDZ10 + self.QDZ11 * dfz) * np.abs(gamma_star))
                          * gamma_star * self.LKZC * zeta_0) * LMUY_star
               * np.sign(VCX) * cos_prime_alpha + self.zeta_8 - 1.0)

        # residual self-aligning couple (4.E36)
        MZR = D_R * self.cos(C_R * self.atan(B_R * alpha_used)) * cos_prime_alpha

        return MZR

    def __trail_main_routine(self, SA: allowableData, FZ: allowableData, P: allowableData, IA: allowableData, VCX: allowableData, VS: allowableData, zeta_5) -> list[allowableData]:
        """Function containing the main calculations for the pneumatic trail. To be used in ``find_trail`` and ``find_trail_pure``."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # scaled nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # normalize pressure and load
        dfz = self.__find_dfz(FZ)
        dpi = self.__find_dpi(P)

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # degressive friction factor (4.E8)
        LMUY_star  = self.__find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.__find_lmu_prime(LMUY_star)

        # stiffness factor (4.E40) TODO QBZ6 changed to QBZ4
        BT = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz ** 2) * (1.0 + self.QBZ5 * np.abs(gamma_star) + self.QBZ4 * gamma_star ** 2) * self.LYKA / LMUY_prime

        # TODO: Figure out which version for BT to use. QBZ4 is not is the book by Pacejka & Besselink, but it is in their 2010 paper
        # BT = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz ** 2) * (1.0 + QBZ4 + QBZ5 * np.abs(gamma_star) + QBZ6 * gamma_star ** 2) * LYKA / LMUY_prime

        # shape factor (4.E41)
        CT = self.QCZ1

        # static peak factor (4.E42)
        DT0 = self.__find_dt0(FZ, dfz, dpi, VCX, FZ0_prime, R0)

        # peak factor (4.E43)
        DT = DT0 * (1.0 + self.QDZ3 * np.abs(gamma_star) + self.QDZ4 * gamma_star ** 2) * zeta_5

        # horizontal shift (4.E35)
        S_HT = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma_star

        # corrected slip angle (4.E34)
        alpha_star = self.__find_alpha_star(SA, VCX)
        alpha_t = alpha_star + S_HT

        # curvature factor (4.E44)
        ET = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz ** 2) * (1.0 + (self.QEZ4 + self.QEZ5 * gamma_star) * np.pi / 2 * self.atan(BT * CT * alpha_t))

        return [BT, CT, DT, ET, alpha_t]

    #------------------------------------------------------------------------------------------------------------------#
    # TURN SLIP EXTENSION

    # TODO
    def __find_zeta_1(self):
        pass

    def __find_zeta_2(self, SA: allowableData, FZ: allowableData, PHI: allowableData) -> allowableData:
        """Peak side force reduction for turn slip extension."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # normalize vertical load
        dfz = self.__find_dfz(FZ)

        # sharpness factor (4.78)
        BYP = self.PDYP1 * (1.0 + self.PDYP2 * dfz) * self.cos(self.atan(self.PDYP3 * self.tan(SA)))

        # second turn slip correction factor (4.77)
        zeta_2 = self.cos(self.atan(BYP * (R0 * np.abs(PHI) + self.PDYP4 * np.sqrt(R0 * np.abs(PHI)))))
        return zeta_2

    def __find_zeta_3(self, PHI: allowableData) -> allowableData:
        """Slope reduction factor for turn slip extension."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        zeta_3 = self.cos(self.atan(self.PKYP1 * R0 ** 2 * PHI ** 2))
        return zeta_3

    def __find_zeta_4(
            self,
            FZ:  allowableData,
            P:   allowableData,
            IA:  allowableData,
            VCX: allowableData,
            VS:  allowableData,
            PHI: allowableData,
            zeta_2,
            angle_unit):

        # unpack tyre properties
        V0 = self.LONGVL

        # assumed that difference between contact patch and wheel center speed is negligible as (eqn 7.4 from Pacejka)
        VX = VCX

        # corrected camber angle
        gamma_star = self.__find_gamma_star(IA)

        # normalize load
        dfz = self.__find_dfz(FZ)

        # difference between camber and turn slip response
        eps_y = self.__find_eps_y(FZ)

        # cornering stiffness
        KYA = self.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # NOTE: this parameter is not explained in the book or paper, but according to Kaustub Ragunathan from
        # IPG-Carmaker it is cornering stiffness for zero camber (via Marco Furlan from MFeval).
        KYAO = self.find_cornering_stiffness(FZ, P, 0.0, PHI, angle_unit)

        # corrected cornering stiffness (4.E39)
        KYA_prime  = KYA  + self.eps_K * np.sign(KYA)
        KYAO_prime = KYAO + self.eps_K * np.sign(KYAO)

        # camber stiffness
        KYCO = self.find_camber_stiffness(FZ, P)

        # spin force stiffness (4.89)
        KYRP0 = KYCO / (1.0 - eps_y)

        # (4.85)
        CHYP = self.PHYP1

        # (4.86)
        DHYP = (self.PHYP2 + self.PHYP3 * dfz) * np.sign(VCX)

        # (4.87)
        EHYP = self.PHYP4

        # (4.88)
        BHYP = KYRP0 / (CHYP * DHYP * KYAO_prime)

        # making side force vanish for large spin (4.80)
        PHI_corr = BHYP * R0 * PHI
        S_HYP = DHYP * self.sin(CHYP * self.atan(PHI_corr - EHYP * (PHI_corr - self.atan(PHI_corr)))) * np.sign(VX)

        # degressive friction factor (4.E8)
        LMUY_star  = self.__find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.__find_lmu_prime(LMUY_star)

        # vertical shift
        S_VYg, _ = self.__find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2)

        # (4.84)
        zeta_4 = 1.0 + S_HYP - S_VYg / KYA_prime
        return zeta_4

    def __find_zeta_5(self, PHI: allowableData) -> allowableData:
        """spin moment decay due to turn slip."""
        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # (4.91)
        zeta_5 = self.cos(self.atan(self.QDTP1 * R0 * PHI))
        return zeta_5

    # TODO
    def __find_zeta_6(self):
        pass

    # TODO
    def __find_zeta_7(self):
        pass

    def __find_zeta_8(self, FZ, P, IA, VS, angle_unit):

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # scaled nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # normalize load
        dfz = self.__find_dfz(FZ)

        # lateral friction coefficient
        mu_y = self.find_mu_y(FZ, P, IA, VS, angle_unit)

        # self aligning moment at vanishing wheel speed (4.95)
        MZP_inf = self.QCRP1 * mu_y * R0 * FZ * np.sqrt(FZ / FZ0_prime) * self.LMP

        # (4.99)
        KZCRO = FZ * R0 * (self.QDZ8 + self.QDZ9 * dfz + (self.QDZ10 + self.QDZ11 * dfz) * np.abs(IA)) * self.LKZC

        # shape factor (4.96)
        CDRP = self.QDRP1

        # peak factor 1 (4.94) -- not corrected to degrees since it depends on pi
        DDRP = MZP_inf / np.sin(0.5 * np.pi * CDRP)

        # curvature factor (4.97)
        EDRP = self.QDRP2

        # stiffness factor (4.98)
        BDRP = KZCRO / (CDRP * DDRP * (1.0 - eps_gamma) + eps_r)

        # peak factor 2 (4.93)
        PHI_corr = BDRP * R0 * PHI
        DRP = DDRP * self.sin(CDRP * np.atan(PHI_corr - EDRP * (PHI_corr - self.atan(PHI_corr))))

        # final correction (4.92)
        zeta_8 = 1.0 + DRP
        return zeta_8
