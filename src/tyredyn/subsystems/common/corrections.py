from tyredyn.types.aliases import SignalLike, NumberLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase
import numpy as np

class Corrections(SubSystemBase):
    """
    Module containing common correction factors for the magic formula, such as ``alpha_star``, ``LMU_star``, etc.
    """

    def _connect(self, model):
        self._normalize = model.normalize

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

    def _find_epsilon_gamma(
            self,
            dfz: SignalLike
    ) -> SignalLike:
        """Returns the camber reduction factor for turn slip."""

        # camber reduction factor (4.90)
        eps_gamma = self.PECP1 * (1.0 - self.PECP2 * dfz)
        return eps_gamma

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

        lmu_prime = self._A_mu * LMU_star / (1.0 + (self._A_mu - 1.0) * LMU_star)
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
            LMU_star = LMU / (1.0 + self._LMUV * VS / V0)
        else:
            LMU_star = LMU
        return LMU_star

    def _find_vc_prime(
            self,
            VC: SignalLike
    ) -> SignalLike:
        """Returns the singularity-protected contact patch speed."""

        # corrected wheel center speed (4.E6a)
        VC_sign = self.signals._replace_value(np.sign(VC), target_sig=VC, target_val=0.0, new_val=1.0)
        VC_prime = VC + self._eps_V * VC_sign

        # NOTE: the book .... ??? TODO

        return VC_prime
