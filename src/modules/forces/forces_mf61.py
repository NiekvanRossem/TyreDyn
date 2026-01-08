from src.utils.formatting import SignalLike, AngleUnit
from typing import Literal
import numpy as np

class ForcesMF6x:
    """
    Forces module for the MF 6.1 and MF 6.2 tyre models.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` or ``MF62`` class."""
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

    def _find_fx_pure(
            self,
            *,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the longitudinal force for pure slip conditions.

        Parameters
        ----------
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        FX0 : SignalLike
            Longitudinal force for pure slip.
        """

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=0.0, SL=SL, VX=VX)

        # reference speed
        V0 = self.LONGVL

        # turn slip correction
        if self._use_turn_slip:
            PHI    = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_1 = self.turn_slip._find_zeta_1(SL=SL, FZ=FZ, PHI=PHI)
        else:
            zeta_1 = self.zeta_default

        # _normalize inputs
        dfz = self.normalize._find_dfz(FZ)

        # composite friction scaling factor (4.E7)
        LMUX_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUX)

        # degressive friction factor (4.E8)
        LMUX_prime = self.correction._find_lmu_prime(LMUX_star)

        # horizontal shift (4.E17)
        S_HX = (self.PHX1 + self.PHX2 * dfz) * self.LHX

        # low speed correction
        S_HX = self.normalize._correct_signal(S_HX, correction_factor=self.smooth_correction, helper_sig=np.abs(VX),
                                              threshold=self.VXLOW, method="<")

        # vertical shift (4.E18)
        S_VX = FZ * (self.PVX1 + self.PVX2 * dfz) * self.LVX * LMUX_prime * zeta_1

        # low speed correction
        S_VX = self.normalize._correct_signal(S_VX, correction_factor=self.smooth_correction, helper_sig=np.abs(VX),
                                              threshold=self.VXLOW, method="<")

        # corrected slip ratio (4.E10)
        kappa_x = SL + S_HX

        # shape factor (4.E11)
        CX = self.PCX1 * self.LCX

        # friction coefficient (4.E13)
        mu_x = self.friction._find_mu_x(SA=0.0, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

        # peak factor (4.E12)
        DX = mu_x * FZ * zeta_1

        # curvature factor (4.E14)
        EX = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz ** 2) * (1.0 - self.PEX4 * np.sign(kappa_x)) * self.LEX

        # slip stiffness (4.E15)
        KXK = self.gradient._find_slip_stiffness(FZ=FZ, P=P)

        # stiffness factor (4.E16)
        BX = KXK / (CX * DX + self._eps_x)

        # Longitudinal force (4.E9) -- slip ratio trig functions do not get corrected to degrees
        FX0 = DX * np.sin(CX * self.atan(BX * kappa_x - EX * (BX * kappa_x - np.atan2(BX * kappa_x, 1)))) + S_VX

        # flip sign for negative speeds (only if alpha_star is used)
        FX0 = self.normalize._flip_negative(FX0, helper_sig=VX) if self._use_alpha_star else FX0

        return FX0

    def _find_fx_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the longitudinal force for combined slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        FX : SignalLike
            Longitudinal force for combined slip conditions.
        """

        # find other velocity components
        VCX = VX

        # normalized vertical load
        dfz = self.normalize._find_dfz(FZ)

        # horizontal shift (4.E57)
        S_HXA = self.RHX1

        # corrected slip angles (4.E53)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)
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
        FX0 = self._find_fx_pure(SL=SL, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # longitudinal force for combined slip (4.E50)
        FX = FX0 * GXA
        return FX

    #------------------------------------------------------------------------------------------------------------------#
    # LATERAL FORCES

    def _find_fy_pure(
            self,
            *,
            SA:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the side force for pure slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        FY0 : SignalLike
            Side force for pure slip conditions.
        """

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=SA, SL=0.0, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI    = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_0 = 0.0  # (4.83)
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
            zeta_4 = self.turn_slip._find_zeta_4(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VCX=VCX, VS=VS, PHI=PHI,
                                                 zeta_2=zeta_2)
        else:
            zeta_0 = self.zeta_default
            zeta_2 = self.zeta_default
            zeta_4 = self.zeta_default

        # find normalized load and pressure
        dfz = self.normalize._find_dfz(FZ)

        # allows for large slip angles and reverse running (4.E3)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)

        # for spin due to camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # reference speed
        V0 = self.LONGVL

        # composite friction scaling factor (4.E7)
        LMUY_star = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)

        # degressive friction factor (4.E8)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # cornering stiffness (4.E25)
        KYA = self.gradient._find_cornering_stiffness(SA=SA, SL=0.0, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # camber stiffness (4.E30)
        KYCO = self.gradient._find_camber_stiffness(FZ=FZ, P=P)

        # vertical shifts (4.E29)
        S_VY, S_VYg = self.common._find_s_vy(FZ=FZ, VX=VX, dfz=dfz, gamma_star=gamma_star, LMUY_prime=LMUY_prime,
                                             zeta_2=zeta_2)

        # horizontal shift (4.E27)
        S_HY = self.common._find_s_hy(VX=VX, dfz=dfz, KYA=KYA, KYCO=KYCO, gamma_star=gamma_star, S_VYg=S_VYg,
                                      zeta_0=zeta_0, zeta_4=zeta_4)

        # corrected slip angle (4.E20)
        alpha_y = alpha_star + S_HY

        # shape factor (4.E21)
        CY = self.common._find_cy()

        # friction coefficient (4.E23)
        mu_y = self.friction._find_mu_y(SA=SA, SL=0.0, FZ=FZ, P=P, IA=IA, VX=VX)

        # peak factor (4.E22)
        DY = self.common._find_dy(mu_y=mu_y, FZ=FZ, zeta_2=zeta_2)

        # curvature factor (4.E24)
        EY = (self.PEY1 + self.PEY2 * dfz) * (1.0 + self.PEY5 * gamma_star ** 2 - (self.PEY3 + self.PEY4 * gamma_star)
                                              * np.sign(alpha_y)) * self.LEY

        # stiffness factor (4.E26)
        BY = self.common._find_by(FZ=FZ, KYA=KYA, CY=CY, DY=DY)

        # lateral force (4.E19)
        FY0 = DY * self.sin(CY * self.atan(BY * alpha_y - EY * (BY * alpha_y - self.atan(BY * alpha_y)))) + S_VY

        # flip sign for negative speeds (only if alpha_star is used)
        FY0 = self.normalize._flip_negative(FY0, helper_sig=VX) if self._use_alpha_star else FY0

        return FY0

    def _find_fy_combined(
            self,
            *,
            SA:   SignalLike,
            SL:   SignalLike,
            FZ:   SignalLike,
            N:    SignalLike,
            P:    SignalLike = None,
            IA:   SignalLike = 0.0,
            VX:   SignalLike = None,
            PHIT: SignalLike = 0.0
    ) -> SignalLike:
        """
        Returns the side force for combined slip conditions.

        Parameters
        ----------
        SA : SignalLike
            Slip angle.
        SL : SignalLike
            Slip ratio.
        FZ : SignalLike
            Vertical load.
        N : SignalLike, optional
            Angular speed of the wheel (will be calculated from ``VX`` and ``SL`` if not specified).
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        IA : SignalLike, optional
            Inclination angle with respect to the ground plane (will default to ``0.0`` if not specified).
        VX : SignalLike, optional
            Contact patch longitudinal speed (will default to ``LONGVL`` if not specified).
        PHIT : SignalLike, optional
            Turn slip (will default to ``0.0`` if not specified).

        Returns
        -------
        FY : SignalLike
            Side force for combined slip conditions.
        """

        # find other velocity components
        VS, VC = self.normalize._find_speeds(SA=SA, SL=SL, VX=VX)
        VCX = VX

        # turn slip correction
        if self._use_turn_slip:
            PHI = self.correction._find_phi(FZ=FZ, N=N, VC=VC, IA=IA, PHIT=PHIT)
            zeta_2 = self.turn_slip._find_zeta_2(SA=SA, FZ=FZ, PHI=PHI)
        else:
            zeta_2 = self.zeta_default

        # normalized vertical load
        dfz = self.normalize._find_dfz(FZ)

        # side force for pure slip
        FY0 = self._find_fy_pure(SA=SA, FZ=FZ, N=N, P=P, IA=IA, VX=VX, PHIT=PHIT)

        # corrected slip angle (4.E53)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)

        # corrected camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # lateral friction coefficient
        mu_y = self.friction._find_mu_y(SA=SA, SL=SL, FZ=FZ, P=P, IA=IA, VX=VX)

        # peak factor (4.E67)
        D_VYK = (mu_y * FZ * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma_star)
                 * self.cos(self.atan(self.RVY4 * alpha_star)) * zeta_2)

        # vertical shift (4.E66) -- slip ratio trig functions do not get corrected to degrees
        S_VYK = D_VYK * np.sin(self.RVY5 * np.atan2(self.RVY6 * SL, 1)) * self.LVYKA

        # combined forces scaling factor
        GYK = self.common._find_gyk(SA=SA, SL=SL, FZ=FZ, IA=IA, VCX=VCX)

        # side force for combined slip (4.E58)
        FY = FY0 * GYK + S_VYK
        return FY
