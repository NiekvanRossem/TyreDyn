from tyredyn.types.aliases import NumberLike, SignalLike
import numpy as np

class RadiusMF6x:
    """Class containing the methods for calculating the various radii and deflection for MF-Tyre 6.1 and MF-Tyre 6.2"""

    def __init__(self, model):
        """Import the properties of the overarching class."""
        self._model     = model

        # helper functions
        self.normalize  = model.normalize
        self.common     = model.common

        # other subsystems used
        self.stiffness  = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_free_radius(
            self,
            *,
            N:  SignalLike,
            R0: NumberLike,
            V0: NumberLike
    ) -> SignalLike:

        # free rolling radius (A3.1)
        R_omega = R0 * (self.Q_RE0 + self.Q_V1 * (R0 * N / V0) ** 2)
        return R_omega

    def _find_effective_radius(
            self,
            *,
            FZ:      SignalLike,
            P:       SignalLike,
            R_omega: SignalLike,
            FZ0:     NumberLike
    ) -> SignalLike:

        # vertical stiffness
        CZ = self.stiffness._find_vertical_stiffness(P)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = R_omega - FZ0 / CZ * (self.FREFF * FZ / FZ0 + self.DREFF * np.atan2(self.BREFF * FZ / FZ0, 1))
        return RE

    def _find_qfz1(self, *, CZ0, R0, FZ0):
        """Find Q_FZ1 from CZ0, as it generally is not specified in a standard TIR file."""

        # (A3.4 from the 2012 book)
        Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)
        return Q_FZ1

    def _converge_loaded_radius(
            self,
            *,
            FX:      SignalLike,
            FY:      SignalLike,
            FZ:      SignalLike,
            N:       SignalLike,
            dpi:     SignalLike,
            IA:      SignalLike,
            R_omega: SignalLike,
            Q_FZ1:   SignalLike,
            FZ0:     NumberLike,
            R0:      NumberLike,
            V0:      NumberLike
    ) -> SignalLike:
        """
        Returns the true loaded radius, for which ``FZ`` matches the estimated ``FZ`` calculated from the input state
        and forces.
        """

        maxiter = 100 # TODO: make adjustable setting
        tolx = 1e-5

        # set bounds for optimization
        RL_lower = 0.95 * R_omega
        RL_upper = R_omega
        RL_newest = None

        # evaluate function for initial bounds
        y_lower = self._find_fz(FX=FX, FY=FY, RL=RL_lower, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0, V0=V0,
                                 Q_FZ1=Q_FZ1) - FZ
        y_upper = self._find_fz(FX=FX, FY=FY, RL=RL_upper, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0, V0=V0,
                                 Q_FZ1=Q_FZ1) - FZ

        # counter TODO: add convergence flags
        counter = 0

        # perform optimization
        for iteration in range(maxiter):

            # make new guess for RL
            RL_newest = RL_upper - (RL_upper - RL_lower) / (y_upper - y_lower) * y_upper
            y_newest = self._find_fz(FX=FX, FY=FY, RL=RL_newest, R_omega=R_omega, N=N, dpi=dpi, IA=IA, FZ0=FZ0, R0=R0,
                                      V0=V0, Q_FZ1=Q_FZ1) - FZ

            # update values
            y_upper = y_newest
            RL_upper = RL_newest

            # check if all values have converged
            error = abs(y_newest).max()
            if error < tolx * FZ0:
                break
            if counter == maxiter - 1:
                warnings.warn(f"Maximum number of iterations reached. No solution for the loaded radius found. Final "
                              f"error is {error:.6e}")
            counter += 1

        return RL_newest

    def _find_fz(self, *args, **kwargs):
        raise NotImplementedError