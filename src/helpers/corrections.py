from src.utils.misc import allowableData
import numpy as np

class CorrectionsMF61:
    """
    Module containing common correction factors for the magic formula, such as ``alpha_star``, ``LMU_star``, etc.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_alpha_star(self, SA: allowableData, VCX: allowableData) -> allowableData:
        """Finds the corrected slip angle."""

        # allows for large slip angles and reverse running (4.E3)
        if self._use_alpha_star:
            alpha_star = self.tan(SA) * np.sign(VCX)
        else:
            alpha_star = SA
        return alpha_star

    def _find_gamma_star(self, IA: allowableData) -> allowableData:
        """Finds the corrected inclination angle."""

        # (4.E4)
        if self._use_gamma_star:
            gamma_star = self.sin(IA)
        else:
            gamma_star = IA
        return gamma_star

    def _find_lmu_prime(self, LMU_star: allowableData) -> allowableData:
        """Finds the composite friction scaling factor."""

        lmu_prime = self.A_mu * LMU_star / (1.0 + (self.A_mu - 1.0) * LMU_star)
        return lmu_prime

    def _find_lmu_star(self, VS: allowableData, V0: float, LMU: float) -> allowableData:
        """Finds the composite friction scaling factor, corrected for slip speed."""

        # (4.E7)
        if self._use_lmu_star:
            LMU_star = LMU / (1.0 + self.LMUV * VS / V0)
        else:
            LMU_star = LMU
        return LMU_star

    def _find_cos_prime_alpha(self, VC: allowableData, VCX: allowableData) -> allowableData:
        """Finds the correction factor for cosine terms when dealing with large slip angles."""

        # corrected wheel center speed (4.E6a)
        VC_prime = VC + self.eps_V

        # cosine correction (4.E6)
        cos_prime_alpha = VCX / VC_prime
        return cos_prime_alpha
