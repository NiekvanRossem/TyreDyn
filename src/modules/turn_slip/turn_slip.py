from src.utils.misc import allowableData
import numpy as np
from typing import Literal

# TODO: when expanding this to different tyre models, figure out if BY, CY, etc have to be switched based on FITTYP.

class TurnSlip:
    """
    Module containing the turn slip extension functions.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # helper functions
        self.common     = model.common
        self.correction = model.correction
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    # TODO
    def __find_zeta_1(self):
        pass

    def __find_zeta_2(self, SA: allowableData, FZ: allowableData, PHI: allowableData) -> allowableData:
        """Peak side force reduction for turn slip extension."""

        # unpack tyre properties
        R0 = self.UNLOADED_RADIUS

        # normalize vertical load
        dfz = self.normalize.__find_dfz(FZ)

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
        gamma_star = self.correction.__find_gamma_star(IA)

        # normalize load
        dfz = self.normalize.__find_dfz(FZ)

        # difference between camber and turn slip response
        eps_y = self.common.__find_eps_y(FZ)

        # cornering stiffness
        KYA = self.gradients.find_cornering_stiffness(FZ, P, IA, PHI, angle_unit)

        # NOTE: this parameter is not explained in the book or paper, but according to Kaustub Ragunathan from
        # IPG-Carmaker it is cornering stiffness for zero camber (via Marco Furlan from MFeval).
        KYAO = self.gradients.find_cornering_stiffness(FZ, P, 0.0, PHI, angle_unit)

        # corrected cornering stiffness (4.E39)
        KYA_prime  = KYA  + self.eps_K * np.sign(KYA)
        KYAO_prime = KYAO + self.eps_K * np.sign(KYAO)

        # camber stiffness
        KYCO = self.gradients.find_camber_stiffness(FZ, P)

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
        LMUY_star  = self.correction.__find_lmu_star(VS, V0, self.LMUY)
        LMUY_prime = self.correction.__find_lmu_prime(LMUY_star)

        # vertical shift
        S_VYg, _ = self.common.__find_s_vy(FZ, dfz, gamma_star, LMUY_prime, zeta_2)

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
        dfz = self.normalize.__find_dfz(FZ)

        # lateral friction coefficient
        mu_y = self.friction.find_mu_y(FZ, P, IA, VS, angle_unit)

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
