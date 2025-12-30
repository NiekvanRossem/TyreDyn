from src.utils.formatting import SignalLike, NumberLike
import numpy as np

class CorrectionsMF61:
    """
    Module containing common correction factors for the magic formula, such as ``alpha_star``, ``LMU_star``, etc.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model
        self.normalize = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_alpha_star(
            self,
            *,
            SA:  SignalLike,
            VCX: SignalLike
    ) -> SignalLike:
        """Finds the corrected slip angle."""

        # allows for large slip angles and reverse running (4.E3)
        if self._use_alpha_star:
            alpha_star = self.tan(SA) * np.sign(VCX)
        else:
            alpha_star = SA
        return alpha_star

    def _find_gamma_star(
            self,
            IA: SignalLike
    ) -> SignalLike:
        """Finds the corrected inclination angle."""

        # (4.E4)
        if self._use_gamma_star:
            gamma_star = self.sin(IA)
        else:
            gamma_star = IA
        return gamma_star

    def _find_lmu_prime(
            self,
            LMU_star: SignalLike
    ) -> SignalLike:
        """Finds the composite friction scaling factor."""

        lmu_prime = self.A_mu * LMU_star / (1.0 + (self.A_mu - 1.0) * LMU_star)
        return lmu_prime

    def _find_lmu_star(
            self,
            *,
            VS:  SignalLike,
            V0:  NumberLike,
            LMU: NumberLike
    ) -> SignalLike:
        """Finds the composite friction scaling factor, corrected for slip speed."""

        # (4.E7)
        if self._use_lmu_star:
            LMU_star = LMU / (1.0 + self.LMUV * VS / V0)
        else:
            LMU_star = LMU
        return LMU_star

    def _find_cos_prime_alpha(
            self,
            *,
            VC:  SignalLike,
            VCX: SignalLike
    ) -> SignalLike:
        """Finds the correction factor for cosine terms when dealing with large slip angles."""

        # corrected wheel center speed (4.E6a)
        VC_prime = self._find_vc_prime(VC)

        # cosine correction (4.E6)
        cos_prime_alpha = VCX / VC_prime
        return cos_prime_alpha

    def _find_vc_prime(self, VC: SignalLike) -> SignalLike:
        """Returns the singularity-protected contact patch speed."""

        # corrected wheel center speed (4.E6a)
        VC_prime = VC + self.eps_V

        return VC_prime

    def _find_phi(
            self,
            *,
            FZ:   SignalLike,
            N:    SignalLike,
            VC:   SignalLike,
            IA:   SignalLike,
            PHIT: SignalLike) -> SignalLike:
        """Returns the total spin of the tyre."""

        # normalize load
        dfz = self.normalize.find_dfz(FZ)

        # singularity-protected speed
        VC_prime = self._find_vc_prime(VC)

        # find the total spin velocity (4.75)
        psi_dot = - PHIT / VC_prime

        # camber reduction factor
        eps_gamma = self._find_epsilon_gamma(dfz)

        # total tyre spin (4.76)
        PHI = (1.0 / VC_prime) * (psi_dot - (1.0 - eps_gamma) * N * self.sin(IA))
        return PHI

    def _find_epsilon_gamma(self, dfz: SignalLike) -> SignalLike:
        """Returns the camber reduction factor for turn slip."""

        # camber reduction factor (4.90)
        eps_gamma = self.PECP1 * (1.0 - self.PECP2 * dfz)
        return eps_gamma