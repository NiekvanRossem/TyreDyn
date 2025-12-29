from src.utils.formatting import SignalLike, AngleUnit, NumberLike
from typing import Union, TypeAlias
import numpy as np


class CommonMF61:
    """
    Module containing functions used in multiple other modules of MF 6.1.
    """

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_by(
            self,
            *,
            FZ:  SignalLike,
            KYA: SignalLike,
            CY:  SignalLike,
            DY:  SignalLike
    ) -> SignalLike:
        """Finds the stiffness factor for the side force. Used in ``ForcesMF61`` and ``MomentsMF61``."""

        # side force stiffness factor (4.E26)
        eps_y = self._find_eps_y(FZ)
        BY = KYA / (CY * DY + eps_y)
        return BY

    def _find_cy(self) -> SignalLike:
        """Finds the shape factor for the side force. Used in ``ForcesMF61`` and ``MomentsMF61``."""

        # (4.E21)
        CY = self.PCY1 * self.LCY
        return CY

    def _find_dt0(
            self,
            *,
            FZ:         SignalLike,
            dfz:        SignalLike,
            dpi:        SignalLike,
            VCX:        SignalLike,
            FZ0_prime:  SignalLike,
            R0:         NumberLike
    ) -> SignalLike:
        """Finds the static peak factor. Used in ``TrailMF61``."""

        # (4.E42) TODO
        DT0 = FZ * (R0 / FZ0_prime) * (self.QDZ1 + self.QDZ2 * dfz) * (1.0 - self.PPZ1 * dpi) * self.LTR * np.sign(VCX)
        return DT0

    @staticmethod
    def _find_dy(
            *,
            mu_y:   SignalLike,
            FZ:     SignalLike,
            zeta_2: SignalLike
    ) -> SignalLike:
        """Finds the peak factor for the side force. Used in ``ForcesMF61`` and ``MomentsMF61``."""

        # (4.E22)
        DY = mu_y * FZ * zeta_2
        return DY

    def _find_eps_y(
            self,
            FZ: SignalLike
    ) -> SignalLike:
        """Difference between camber and turn slip response. Used internally and in ``TurnSlip``."""

        if self._use_turn_slip:

            # normalize load
            dfz = self.normalize._find_dfz(FZ)

            # difference between camber and turn slip response (4.90)
            eps_y = self.PECP1 * (1.0 + self.PECP2 * dfz)

        else:
            eps_y = 1e-6

        return eps_y

    def _find_gyk(
            self,
            *,
            SA:  SignalLike,
            SL:  SignalLike,
            FZ:  SignalLike,
            IA:  SignalLike,
            VCX: SignalLike
    ) -> SignalLike:
        """Returns the side force scaling factor for combined slip conditions."""

        # corrected slip angle (4.E53)
        alpha_star = self.correction._find_alpha_star(SA=SA, VCX=VCX)

        # corrected camber angle (4.E4)
        gamma_star = self.correction._find_gamma_star(IA)

        # normalize load
        dfz = self.normalize._find_dfz(FZ)

        # stiffness factor (4.E62)
        B_YK = (self.RBY1 + self.RBY4 * gamma_star ** 2) * self.cos(self.atan(self.RBY2 * (alpha_star - self.RBY3))) * self.LYKA

        # shape factor (4.E63)
        C_YK = self.RCY1

        # curvature factor (4.E64)
        E_YK = self.REY1 + self.REY2 * dfz

        # horizontal shift (4.E65)
        S_HYK = self.RHY1 + self.RHY2 * dfz

        # corrected slip ratio (4.E61)
        kappa_s = SL + S_HYK

        # static correction (4.E60) -- slip ratio trig functions do not get corrected to degrees
        GYKO = np.cos(C_YK * np.atan2(B_YK * S_HYK - E_YK * (B_YK * S_HYK - np.atan2(B_YK * S_HYK, 1)), 1))

        # force correction (4.E59) -- slip ratio trig functions do not get corrected to degrees
        GYK = np.cos(C_YK * np.atan2(B_YK * kappa_s - E_YK * (B_YK * kappa_s - np.atan2(B_YK * kappa_s, 1)), 1)) / GYKO
        return GYK

    def _find_s_hy(
            self,
            *,
            dfz:        SignalLike,
            KYA:        SignalLike,
            KYCO:       SignalLike,
            gamma_star: SignalLike,
            S_VYg:      SignalLike,
            zeta_0:     SignalLike,
            zeta_4:     SignalLike,
    ) -> SignalLike:
        """Finds the horizontal shift for the side force. Used in ``ForcesMF61`` and ``MomentsMF61``."""

        # (4.E27)
        S_HY = ((self.PHY1 + self.PHY2 * dfz) * self.LHY + (KYCO * gamma_star - S_VYg)
                / (KYA + self.eps_kappa) * zeta_0 + zeta_4 - 1.0)
        return S_HY

    def _find_s_vy(
            self,
            *,
            FZ:         SignalLike,
            dfz:        SignalLike,
            gamma_star: SignalLike,
            LMUY_prime: SignalLike,
            zeta_2:     SignalLike
    ) -> SignalLike:
        """Finds the vertical shifts for the side force. Used in ``ForcesMF61``, ``MomentsMF61``, and ``TurnSlip``."""

        # vertical shift due to camber (4.E28)
        S_VYg = FZ * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta_2

        # total vertical shift (4.E29)
        S_VY = FZ * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta_2 + S_VYg
        return S_VY, S_VYg
