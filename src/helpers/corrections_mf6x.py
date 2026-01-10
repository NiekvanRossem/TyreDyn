from src.utils.formatting import SignalLike, NumberLike
import numpy as np

class CorrectionsMF6x:
    """
    Module containing common correction factors for the magic formula, such as ``alpha_star``, ``LMU_star``, etc.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model
        self._normalize = model.normalize

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

    def _find_phi(
            self,
            *,
            FZ:   SignalLike,
            N:    SignalLike,
            VC:   SignalLike,
            IA:   SignalLike,
            PHIT: SignalLike
    ) -> SignalLike:
        """Returns the total spin of the tyre."""

        # _normalize load
        dfz = self._normalize._find_dfz(FZ)

        # singularity-protected speed # TODO: MFeval uses VC_prime = V
        VC_prime = self._find_vc_prime(VC)

        # find the total spin velocity (4.75)
        psi_dot = - PHIT / VC_prime

        # camber reduction factor
        eps_gamma = self._find_epsilon_gamma(dfz)

        # total tyre spin (4.76)
        PHI = (1.0 / VC_prime) * (psi_dot - (1.0 - eps_gamma) * N * self.sin(IA))
        return PHI

    def _find_vc_prime(
            self,
            VC: SignalLike
    ) -> SignalLike:
        """Returns the singularity-protected contact patch speed."""

        # corrected wheel center speed (4.E6a)
        VC_sign = self.normalize._replace_value(np.sign(VC), target_sig=VC, target_val=0.0, new_val=1.0)
        VC_prime = VC + self._eps_V * VC_sign

        # NOTE: the book .... ???

        return VC_prime

    def _find_smooth_reduction(
            self,
            VX : SignalLike
    ) -> SignalLike:

        # smooth reduction factor for low speed correction
        smooth_reduction = 1.0 - 0.5 * (1.0 + np.cos(np.pi * VX / self.VXLOW))
        return smooth_reduction
