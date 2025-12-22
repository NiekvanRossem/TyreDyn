from src.utils.misc import allowableData
from typing import Literal
import numpy as np

class ForcesMF61:
    """
    Forces module for MF 6.1
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize
        self.common     = model.common

        # other modules used
        self.friction   = model.friction
        self.gradient   = model.gradient
        self.turn_slip  = model.turn_slip

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    #------------------------------------------------------------------------------------------------------------------#
    # LONGITUDINAL FORCES

    def find_fx_pure(
            self,
            SL:  allowableData,
            FZ:  allowableData,
            P:   allowableData = None,
            IA:  allowableData = 0.0,
            VS:  allowableData = 0.0,
            PHI: allowableData = None,
            angle_unit: Literal["deg", "rad"] = "rad") -> allowableData:
        """
        Returns the longitudinal force for pure slip conditions.

        Parameters
        ----------
        SL : allowableData
            Slip ratio.
        FZ : allowableData
            Vertical load.
        P : allowableData, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : allowableData, optional
            Inclination angle with respect to the ground plane (will default to zero if not specified).
        VS : allowableData, optional
            Slip speed magnitude (will default to zero if not specified).
        PHI : allowableData, optional
            Turn slip (will default to zero if not specified).
        angle_unit : string, optional
            Unit of the signals indicating an angle. Set to ``"deg"`` if your input arrays are specified in degrees.

        Returns
        -------
        FX : allowableData
            Longitudinal force for pure slip.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        PHI = 0.0 if PHI is None else PHI

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SL, FZ, P, IA, VS = self._format_check([SL, FZ, P, IA, VS])

        # perform limit checks
        if self._check_limits:
            self._limit_check(None, SL, FZ, P, IA)

        # correct angle if mismatched between input array and TIR file
        IA, angle_unit = self._angle_unit_check(IA, angle_unit)

        # reference speed
        V0 = self.LONGVL

        # turn slip correction
        if self._use_turn_slip:
            zeta_1 = self.turn_slip._find_zeta_1() # TODO
        else:
            zeta_1 = self.zeta_default

        # normalize inputs
        dfz = self.normalize._find_dfz(FZ)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.correction._find_lmu_star(VS, V0, self.LMUX)

        # degressive friction factor (4.E8)
        LMUX_prime = self.correction._find_lmu_prime(LMUX_star)

        # horizontal shift (4.E17)
        S_HX = (self.PHX1 + self.PHX2 * dfz) * self.LHX

        # vertical shift (4.E18)
        S_VX = FZ * (self.PVX1 + self.PVX2 * dfz) * self.LVX * LMUX_prime * zeta_1

        # corrected slip ratio (4.E10)
        kappa_x = SL + S_HX

        # shape factor (4.E11)
        C_X = self.PCX1 * self.LCX

        # friction coefficient (4.E13)
        mu_x = self.friction.find_mu_x(FZ, P, IA, VS, angle_unit)

        # peak factor (4.E12)
        D_X = mu_x * FZ * zeta_1

        # curvature factor (4.E14)
        E_X = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) * (1.0 - self.PEX4 * np.sign(kappa_x)) * self.LEX

        # slip stiffness (4.E15)
        KXK = self.gradient.find_slip_stiffness(FZ, P)

        # stiffness factor (4.E16)
        B_X = KXK / (C_X * D_X + self.eps_x)

        # Longitudinal force (4.E9) -- slip ratio trig functions do not get corrected to degrees
        FX = D_X * np.sin(C_X * self.atan(B_X * kappa_x - E_X * (B_X * kappa_x - np.atan2(B_X * kappa_x, 1)))) + S_VX

        return FX

    def find_fx_combined(
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
        Returns the longitudinal force for combined slip conditions.

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
        FX : allowableData
            Longitudinal force for combined slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS = self._format_check([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # normalized vertical load
        dfz = self.normalize._find_dfz(FZ)

        # horizontal shift (4.E57)
        S_HXA = self.RHX1

        # corrected slip angles (4.E53)
        alpha_star = self.correction._find_alpha_star(SA, VCX)
        alpha_s = alpha_star + S_HXA

        # corrected camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

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
        FX0 = self.find_fx_pure(SL, FZ, P, IA, VS, PHI, angle_unit)

        # longitudinal force for combined slip (4.E50)
        FX = FX0 * GXA
        return FX

    #------------------------------------------------------------------------------------------------------------------#
    # LATERAL FORCES

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
        Returns the side force for pure slip conditions.

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
        FY : allowableData
            Side force for pure slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX
        PHI = 0.0 if PHI is None else PHI

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, FZ, P, IA, VS, VCX = self._format_check([SA, FZ, P, IA, VS, VCX])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            zeta_0 = 0.0  # (4.83)
            zeta_2 = self.turn_slip._find_zeta_2(SA, FZ, PHI)
            zeta_4 = self.turn_slip._find_zeta_4(FZ, P, IA, VCX, VS, PHI, zeta_2, angle_unit)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default

        # find normalized load and pressure
        dfz = self.normalize._find_dfz(FZ)

        # allows for large slip angles and reverse running (4.E3)
        alpha_star = self.correction._find_alpha_star(SA, VCX)

        # for spin due to camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # reference speed
        V0 = self.LONGVL

        # composite friction scaling factor (4.E7)
        LMUY_star = self.correction._find_lmu_star(VS, V0, self.LMUY)

        # degressive friction factor (4.E8)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # cornering stiffness (4.E25)
        KYA = self.gradient.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # camber stiffness (4.E30)
        KYCO = self.gradient.find_camber_stiffness(FZ, P)

        # vertical shifts (4.E29)
        S_VY, S_VYg = self.common._find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.common._find_s_hy(dfz, KYA, KYCO, gamma_star, S_VYg, zeta_0, zeta_4)

        # corrected slip angle (4.E20)
        alpha_y = alpha_star + S_HY

        # shape factor (4.E21)
        C_Y = self.common._find_cy()

        # friction coefficient (4.E23)
        mu_y = self.friction.find_mu_y(FZ, P, IA, VS, angle_unit)

        # peak factor (4.E22)
        D_Y = self.common._find_dy(mu_y, FZ, zeta_2)

        # curvature factor (4.E24)
        E_Y = (self.PEY1 + self.PEY2 * dfz) * (1.0 + self.PEY5 * gamma_star ** 2 - (self.PEY3 + self.PEY4 * gamma_star)
                                               * np.sign(alpha_y)) * self.LEY

        # stiffness factor (4.E26)
        B_Y = self.common._find_by(FZ, KYA, C_Y, D_Y)

        # lateral force (4.E19)
        FY = D_Y * self.sin(C_Y * self.atan(B_Y * alpha_y - E_Y * (B_Y * alpha_y - self.atan(B_Y * alpha_y)))) + S_VY

        return FY

    def find_fy_combined(
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
        Returns the side force for combined slip conditions.

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
        FY : allowableData
            Side force for combined slip conditions.
        """

        # set default values for optional arguments
        P   = self.INFLPRES if P is None else P
        VCX = self.LONGVL if VCX is None else VCX
        PHI = 0.0 if PHI is None else PHI

        # check if arrays have the right dimension, and flatten if needed
        if self._check_format:
            SA, SL, FZ, P, IA, VCX, VS = self._format_check([SA, SL, FZ, P, IA, VCX, VS])

        # correct angle if mismatched between input array and TIR file
        [SA, IA], angle_unit = self._angle_unit_check([SA, IA], angle_unit)

        # turn slip correction
        if self._use_turn_slip:
            zeta_2 = self.turn_slip._find_zeta_2(SA, FZ, PHI)
        else:
            zeta_2 = self.zeta_default

        # normalized vertical load
        dfz = self.normalize._find_dfz(FZ)

        # corrected slip angle (4.E53)
        alpha_star = self.correction._find_alpha_star(SA, VCX)

        # corrected camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # side force for pure slip
        FY0 = self.find_fy_pure(SA, FZ, P, IA, VCX, VS, PHI, angle_unit)

        # lateral friction coefficient
        mu_y = self.friction.find_mu_y(FZ, P, IA, VS, angle_unit)

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
