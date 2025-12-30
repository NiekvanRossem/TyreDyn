from src.utils.formatting import SignalLike, AngleUnit, NumberLike
import numpy as np
from typing import Literal
from src.modules.gradients.gradients_mf61 import GradientsMF61

# TODO: when expanding this to different tyre tyre_models, figure out if BY, CY, etc have to be switched based on FITTYP.

class TurnSlip:
    """
    Module containing the turn slip extension functions. Turn slip is only available for MF 6.0 or newer.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.common     = model.common
        self.correction = model.correction
        self.normalize  = model.normalize

        # other modules
        self.friction   = model.friction

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    # NEEDS PHI
    def _find_zeta_1(
            self,
            *,
            SL: SignalLike,
            FZ: SignalLike,
            PHI: SignalLike
    ) -> SignalLike:
        """Returns the turn slip correction that scales the longitudinal force."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # normalize load
        dfz = self.normalize._find_dfz(FZ)

        # stiffness factor (4.106)
        BXP = self.PDXP1 * (1.0 + self.PDXP2 * dfz) * np.cos(np.atan2(self.PDXP3 * SL, 1))

        # factor that decreases FX at increasing spin (4.105)
        zeta_1 = np.cos(np.atan2(BXP * R0 * PHI, 1))
        return zeta_1

    # NEEDS PHI
    def _find_zeta_2(
            self,
            *,
            SA:  SignalLike,
            FZ:  SignalLike,
            PHI: SignalLike
    ) -> SignalLike:
        """Returns the turn slip correction that scales the peak side force."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # normalize vertical load
        dfz = self.normalize._find_dfz(FZ)

        # sharpness factor (4.78)
        BYP = self.PDYP1 * (1.0 + self.PDYP2 * dfz) * self.cos(self.atan(self.PDYP3 * self.tan(SA)))

        # second turn slip correction factor (4.77)
        zeta_2 = self.cos(self.atan(BYP * (R0 * np.abs(PHI) + self.PDYP4 * np.sqrt(R0 * np.abs(PHI)))))
        return zeta_2

    # NEEDS PHI
    def _find_zeta_3(
            self,
            PHI: SignalLike
    ) -> SignalLike:
        """Returns the turn slip correction that scales the cornering stiffness."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        zeta_3 = self.cos(self.atan(self.PKYP1 * R0 ** 2 * PHI ** 2))
        return zeta_3

    # NEEDS PHI
    def _find_zeta_4(
            self,
            *,
            FZ:     SignalLike,
            P:      SignalLike,
            IA:     SignalLike,
            VCX:    SignalLike,
            VS:     SignalLike,
            PHI:    SignalLike,
            zeta_2: SignalLike,
            angle_unit: AngleUnit
    ) -> SignalLike:
        """Returns the turn slip correction that scales the side force."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        V0 = self.LONGVL

        # assumed that difference between contact patch and wheel center speed is negligible as (eqn 7.4 from Pacejka)
        VX = VCX

        # corrected camber angle
        gamma_star = self.correction._find_gamma_star(IA)

        # normalize load
        dfz = self.normalize._find_dfz(FZ)

        # difference between camber and turn slip response
        eps_y = self.common._find_eps_y(FZ)

        # cornering stiffness
        KYA = GradientsMF61.find_cornering_stiffness(self, FZ=FZ, P=P, IA=IA, PHIT=PHI, angle_unit=angle_unit)

        # NOTE: this parameter is not explained in the book or paper, but according to Kaustub Ragunathan from
        # IPG-Carmaker it is cornering stiffness for zero camber (via Marco Furlan from MFeval).
        KYAO = GradientsMF61.find_cornering_stiffness(self, FZ=FZ, P=P, IA=0.0, PHIT=PHI, angle_unit=angle_unit)

        # corrected cornering stiffness (4.E39)
        KYA_prime  = KYA  + self.eps_kappa * np.sign(KYA)
        KYAO_prime = KYAO + self.eps_kappa * np.sign(KYAO)

        # camber stiffness
        KYCO = GradientsMF61.find_camber_stiffness(self, FZ=FZ)

        # spin force stiffness (4.89)
        KYRP0 = KYCO / (1.0 - eps_y)

        # shape factor (4.85)
        CHYP = self.PHYP1

        # peak factor (4.86)
        DHYP = (self.PHYP2 + self.PHYP3 * dfz) * np.sign(VCX)

        # curvature factor (4.87)
        EHYP = self.PHYP4

        # stiffness factor (4.88)
        BHYP = KYRP0 / (CHYP * DHYP * KYAO_prime)

        # making side force vanish for large spin (4.80)
        PHI_corr = BHYP * R0 * PHI
        S_HYP = DHYP * self.sin(CHYP * self.atan(PHI_corr - EHYP * (PHI_corr - self.atan(PHI_corr)))) * np.sign(VX)

        # degressive friction factor (4.E8)
        LMUY_star  = self.correction._find_lmu_star(VS=VS, V0=V0, LMU=self.LMUY)
        LMUY_prime = self.correction._find_lmu_prime(LMUY_star)

        # vertical shift
        _, S_VYg = self.common._find_s_vy(FZ=FZ, dfz=dfz, gamma_star=gamma_star, LMUY_prime=LMUY_prime, zeta_2=zeta_2)

        # (4.84)
        zeta_4 = 1.0 + S_HYP - S_VYg / KYA_prime
        return zeta_4

    # NEEDS PHI
    def _find_zeta_5(
            self,
            PHI: SignalLike
    ) -> SignalLike:
        """Returns the turn slip correction that scales the pneumatic trail."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # (4.91)
        zeta_5 = self.cos(self.atan(self.QDTP1 * R0 * PHI))
        return zeta_5

    # NEEDS PHI
    def _find_zeta_6(
            self,
            PHI: SignalLike
    ) -> SignalLike:
        """Returns the turn slip correction that scales the stiffness for the self-aligning couple."""

        # unpack tyre parameters
        R0 = self.UNLOADED_RADIUS

        # turn slip correction -- trig functions for turn slip do not get corrected to degrees
        zeta_6 = np.cos(np.atan(self.QBRP1 * R0 * PHI)) # TODO: check if PHI and PHI_t are the same
        return zeta_6

    # NEEDS PHI AND PHIT
    def _find_zeta_7(
            self,
            *,
            SA:  SignalLike,
            SL:  SignalLike,
            FZ:  SignalLike,
            P:   SignalLike,
            IA:  SignalLike,
            VCX: SignalLike,
            VS:  SignalLike,
            PHI: SignalLike,
            angle_unit: AngleUnit
    ) -> SignalLike:
        """Returns the turn slip correction that sets the shape factor for the self-aligning couple.

        Parameters
        ----------
        *
        """

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # lateral friction coefficient
        mu_y = self.friction.find_mu_y(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

        # turn slip moment at vanishing speed
        MZP_inf = self.QCRP1 * mu_y * R0 * FZ * np.sqrt(FZ / FZ0_prime) * self.LMP

        # combined slip scaling factor
        GYK = self.common._find_gyk(SA=SA, SL=SL, FZ=FZ, IA=IA, VCX=VCX)

        # turn slip moment at 90 degrees NEEDS PHIT
        MZP_90 = MZP_inf * (2.0 / np.pi) * np.atan2(self.QCRP2 * R0 * np.abs(PHIT), 1) * GYK

        # peak factor NEEDS PHI
        DRP = self.__find_drp(FZ=FZ, P=P, IA=IA, VS=VS, PHI=PHI, R0=R0, FZ0_prime=FZ0_prime, angle_unit=angle_unit)

        # turn slip correction
        zeta_7 = (2.0 / np.pi) * np.acos(MZP_90 / (np.abs(DRP) + self.eps_r))
        return zeta_7

    # NEEDS PHI
    def _find_zeta_8(
            self,
            *,
            FZ:  SignalLike,
            P:   SignalLike,
            IA:  SignalLike,
            VS:  SignalLike,
            PHI: SignalLike,
            angle_unit: AngleUnit
    ) -> SignalLike:
        """Returns the turn slip correction that scales the peak of the residual couple."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # scaled nominal load
        FZ0 = self.FNOMIN
        FZ0_prime = FZ0 * self.LFZO

        # peak factor
        DRP = self.__find_drp(FZ=FZ, P=P, IA=IA, VS=VS, PHI=PHI, R0=R0, FZ0_prime=FZ0_prime, angle_unit=angle_unit)

        # final correction (4.92)
        zeta_8 = 1.0 + DRP
        return zeta_8

    # NEEDS PHI
    def __find_drp(
            self,
            *,
            FZ:  SignalLike,
            P:   SignalLike,
            IA:  SignalLike,
            VS:  SignalLike,
            PHI: SignalLike,
            R0:  SignalLike,
            FZ0_prime: NumberLike,
            angle_unit: AngleUnit
    ) -> SignalLike:
        """Returns the ``DRP`` parameter used by ``zeta_7`` and ``zeta_8``."""
        
        # normalize load
        dfz = self.normalize._find_dfz(FZ)

        # lateral friction coefficient
        mu_y = self.friction.find_mu_y(FZ=FZ, P=P, IA=IA, VS=VS, angle_unit=angle_unit)

        # self aligning moment at vanishing wheel speed (4.95)
        MZP_inf = self.QCRP1 * mu_y * R0 * FZ * np.sqrt(FZ / FZ0_prime) * self.LMP

        # (4.99)
        KZCRO = FZ * R0 * (self.QDZ8 + self.QDZ9 * dfz + (self.QDZ10 + self.QDZ11 * dfz) * np.abs(IA)) * self.LKZC

        # shape factor (4.96)
        CDRP = self.QDRP1

        # peak factor 1 (4.94) -- not corrected to degrees since it depends on pi
        DDRP = MZP_inf / np.sin(0.5 * np.pi * CDRP)

        # camber reduction factor
        eps_gamma = self.correction._find_epsilon_gamma(dfz)

        # stiffness factor
        # NOTE: the equation below is taken from the equation manual, which differs from the one in the book, shown
        # below: (4.98). Via Marco Furlan from MFeval
        # BDRP = KZCRO / (CDRP * DDRP * (1.0 - eps_gamma) + self.eps_r)
        BDRP = KZCRO / (CDRP * DDRP * (1.0 - eps_gamma))

        # peak factor 2
        # NOTE: the equation used is taken from the equation manual, which differs from the one in the book, shown
        # below in the comment (4.93). EDRP is equal to QDRP2 (4.93), but this parameter generally does not exist in TIR
        # files. Via Marco Furlan from MFeval
        # DRP = DDRP * self.sin(CDRP * np.atan(PHI_corr - EDRP * (PHI_corr - self.atan(PHI_corr))))
        PHI_corr = BDRP * R0 * PHI
        DRP = DDRP * np.sin(CDRP * np.atan2(PHI_corr, 1))

        return DRP

