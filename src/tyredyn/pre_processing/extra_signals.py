import numpy as np
from tyredyn.types.aliases import SignalLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase

class ExtraSignals(SubSystemBase):
    """Module containing the methods to calculate dependent input signals."""

    def _connect(self, model):
        self.normalize  = model.normalize
        self.correction = model.correction
        self.radius     = model.radius

    #def __init__(self, model):
    #    """Make the properties of the overarching class and other subsystems available."""
    #    self._model = model

    #    # helper functions
    #    self.normalize  = model.normalize
    #    self.correction = model.correction

    #def __getattr__(self, name):
    #    """Make the tyre coefficients directly available."""
    #    return getattr(self._model, name)

    #------------------------------------------------------------------------------------------------------------------#

    def _find_omega(
            self,
            *,
            SL: SignalLike,
            FZ: SignalLike,
            VX: SignalLike,
            P:  SignalLike,
    ) -> SignalLike:
        """Returns the angular speed of the wheel based on the longitudinal velocity and the slip ratio."""

        # unpack tyre properties
        FZ0 = self.FNOMIN
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL

        # first guess of angular speed based on unloaded radius
        N = VX / R0
        err = 1
        counter = 0

        # converge over N TODO: add convergence flags and filter for arrays
        while np.max(err) > 1e-9: # TODO: make adjustable setting
            counter += 1

            # free rolling radius
            R_omega = self.radius._find_free_radius(N=N, R0=R0, V0=V0)

            # effective radius
            RE = self.radius._find_effective_radius(FZ=FZ, P=P, R_omega=R_omega, FZ0=FZ0)

            # find new angular speed (2.5)
            N_old = N
            N = ((1.0 + SL) * VX) / RE
            err = np.abs(N - N_old)

            if counter > 100:  # TODO: make adjustable setting
                warnings.warn(f"No estimate for the angular speed found! Final error is {err:.3e}")
                break

        return N

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
        dfz = self.normalize._find_dfz(FZ)

        # singularity-protected speed # TODO: MFeval uses VC_prime = V
        VC_prime = self.correction._find_vc_prime(VC)

        # find the total spin velocity (4.75)
        psi_dot = - PHIT / VC_prime

        # camber reduction factor
        eps_gamma = self.correction._find_epsilon_gamma(dfz)

        # total tyre spin (4.76)
        PHI = (1.0 / VC_prime) * (psi_dot - (1.0 - eps_gamma) * N * self.sin(IA))
        return PHI

    def _find_speeds(
            self,
            *,
            SA: SignalLike,
            SL: SignalLike,
            VX: SignalLike
    ) -> SignalLike:
        """Finds the speed of the slip point ``S`` as well as the speed of point ``C``."""

        # longitudinal slip speed (2.11)
        VSX = - SL * VX

        # lateral slip speed (2.12)
        VSY = VX * self.tan(SA)

        # total slip speed (3.39)
        VS = np.sqrt(VSX ** 2 + VSY ** 2)

        # total contact patch speed
        V = np.sqrt(VX ** 2 + VSY ** 2)

        return VS, V
