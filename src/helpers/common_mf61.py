from src.utils.misc import allowableData
import numpy as np

class CommonMF61:
    """
    Module containing functions used in multiple other modules. MF 6.1.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

        # helper functions
        self.correction = model.correction
        self.normalize  = model.normalize

        # CANNOT DEPEND ON TURN SLIP!

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_by(self, FZ: allowableData, KYA: allowableData, CY: allowableData, DY: allowableData) -> allowableData:
        """Finds the stiffness factor for the side force. Used in ``ForcesMF61`` and ``MomentsMF61``."""

        # side force stiffness factor (4.E26)
        eps_y = self._find_eps_y(FZ)
        BY = KYA / (CY * DY + eps_y)
        return BY

    def _find_cy(self) -> allowableData:
        """Finds the shape factor for the side force."""

        # (4.E21)
        CY = self.PCY1 * self.LCY
        return CY

    def _find_dt0(
            self,
            FZ:         allowableData,
            dfz:        allowableData,
            dpi:        allowableData,
            VCX:        allowableData,
            FZ0_prime:  allowableData,
            R0:         Union[int, float]) -> allowableData:
        """Finds the static peak factor."""

        # (4.E42) TODO
        DT0 = FZ * (R0 / FZ0_prime) * (self.QDZ1 + self.QDZ2 * dfz) * (1.0 - self.PPZ1 * dpi) * self.LTR * np.sign(VCX)
        return DT0

    @staticmethod
    def _find_dy(mu_y: allowableData, FZ: allowableData, zeta_2) -> allowableData:
        """Finds the peak factor for the side force."""

        # (4.E22)
        DY = mu_y * FZ * zeta_2
        return DY

    def _find_eps_y(self, FZ):
        """Difference between camber and turn slip response. Used internally and in ``TurnSlip``."""

        if self._use_turn_slip:

            # normalize load
            dfz = self.__find_dfz(FZ)

            # difference between camber and turn slip response (4.90)
            eps_y = self.PECP1 * (1.0 + self.PECP2 * dfz)

        else:
            eps_y = 1e-6

        return eps_y

    def _find_s_hy(self, dfz: allowableData, KYA: allowableData, KYCO: allowableData, gamma_star: allowableData, S_VYg: allowableData, zeta_0, zeta_4) -> allowableData:
        """Finds the horizontal shift for the side force."""

        # (4.E27)
        S_HY = ((self.PHY1 + self.PHY2 * dfz) * self.LHY + (KYCO * gamma_star - S_VYg)
                / (KYA + self.eps_K) * zeta_0 + zeta_4 - 1.0)
        return S_HY

    def _find_s_vy(self, FZ: allowableData, dfz: allowableData, gamma_star: allowableData, LMUY_prime: allowableData, zeta_2) -> allowableData:
        """Finds the vertical shifts for the side force. Used in ``ForcesMF61``, ``MomentsMF61``, and ``TurnSlip``."""

        # vertical shift due to camber (4.E28)
        S_VYg = FZ * (self.PVY3 + self.PVY4 * dfz) * gamma_star * self.LKYC * LMUY_prime * zeta_2

        # total vertical shift (4.E29)
        S_VY = FZ * (self.PVY1 + self.PVY2 * dfz) * self.LVY * LMUY_prime * zeta_2 + S_VYg
        return S_VYg, S_VY
