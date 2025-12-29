from src.utils.formatting import SignalLike
from typing import Literal
import numpy as np
import warnings

# TODO: make extra optimization for finding ``RL`` if ``N`` is not an input

class RadiusMF61:

    def __init__(self, model):
        """Import the properties of the overarching ``MF61`` class."""
        self._model = model

        # helper functions
        self.normalize  = model.normalize

        # other modules used
        self.stiffness  = model.stiffness

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def find_radius(
            self,
            *,
            FX: SignalLike,
            FY: SignalLike,
            FZ: SignalLike,
            N:  SignalLike,
            P:  SignalLike = None,
            **kwargs
    ) -> list[SignalLike]:
        """
        Returns the various radii and deflection of the tyre. Order is ``R_omega``, ``RE``, ``RL``, ``rho``.

        Parameters
        ----------
        FX : SignalLike
            Longitudinal force.
        FY : SignalLike
            Side force.
        FZ : SignalLike
            Vertical load.
        N : SignalLike
            Angular speed of the wheel.
        P : SignalLike, optional
            Tyre pressure (will default to ``INFLPRES`` if not specified).
        kwargs : any, optional
            Allows other arguments to be passed for compatibility with ``MF62``. Arguments passed will not be used.

        Returns
        -------
        R_omega, RE, RL, rho : list[SignalLike]
            Free rolling radius, effective radius, loaded radius, and vertical deflection.
        """

        # set default values for optional arguments
        P = self.INFLPRES if P is None else P

        # check if arrays have the right dimension, and flatten if needed TODO move to outer functions
        if self._check_format:
            FX, FY, FZ, N, P = self._format_check([FX, FY, FZ, N, P])

        # unpack tyre properties
        CZ0 = self.VERTICAL_STIFFNESS
        FZ0 = self.FNOMIN
        R0  = self.UNLOADED_RADIUS
        V0  = self.LONGVL

        # normalize tyre pressure
        dpi = self.normalize._find_dpi(P)

        # free rolling radius (A3.1)
        R_omega = R0 * (self.Q_RE0 + self.Q_V1 * (R0 * N / V0) ** 2)

        # vertical stiffness
        CZ = self.stiffness.find_vertical_stiffness(P)

        # effective radius (A3.6) -- FZ trig functions do not get corrected to degrees
        RE = R_omega - FZ0 / CZ * (self.FREFF * FZ / FZ0 + self.DREFF * np.atan2(self.BREFF * FZ / FZ0, 1))

        # find QFZ1 from CZ0 (A3.4)
        Q_FZ1 = np.sqrt((CZ0 * R0 / FZ0) ** 2 - 4 * self.Q_FZ2)

        # inputs affecting the radius (A3.3) TODO: equation 4.E68 adds extra camber terms to it.
        speed_effect    = self.Q_V2 * np.abs(N) * R0 / V0
        fx_effect       = (self.Q_FCX * FX / FZ0) ** 2
        fy_effect       = (self.Q_FCY * FY / FZ0) ** 2
        pressure_effect = (1.0 + self.PFZ1 * dpi) * FZ0

        # solve via the ABC formula
        A = - self.Q_FZ2 / (R0 ** 2)
        B = - Q_FZ1 / R0
        C = FZ / ((1.0 + speed_effect - fx_effect - fy_effect) * pressure_effect)
        rho = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        # display warning if only imaginary solutions can be found for a datapoint
        check_root = B ** 2 - 4 * A * C
        if not isinstance(check_root, np.ndarray):
            if isinstance(check_root, list):
                check_root = np.array(check_root)
            else:
                check_root = np.array([check_root])
        if any(check_root < 0.0):
            warnings.warn("No real solution found for the tyre deflection!")

        # apply proper limits to avoid dividing by zero
        rho = np.maximum(rho, 1e-6)

        # loaded radius
        RL = R_omega - rho

        return [R_omega, RE, RL, rho]
