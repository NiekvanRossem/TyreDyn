from src.utils.formatting import SignalLike
import numpy as np

# TODO: rename

class Normalize:
    """
    Module containing the functions to _normalize input signals.
    """

    def __init__(self, model):
        """Make the properties of the overarching ``MF61`` class and other modules available."""
        self._model = model

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_dfz(
            self,
            FZ: SignalLike
    ) -> SignalLike:
        """Finds the normalized vertical load."""

        # unpack parameters
        FZO = self.FNOMIN

        # scale nominal load (4.E1)
        FZO_scaled = self.LFZO * FZO

        # find normalized vertical load (4.E2a)
        dfz = (FZ - FZO_scaled) / FZO_scaled

        return dfz

    def _find_dpi(
            self,
            P: SignalLike
    ) -> SignalLike:
        """Finds the normalized tyre pressure."""

        # extract parameters
        PO = self.NOMPRES

        # normalized pressure (4.E2b)
        dpi = (P - PO) / PO
        return dpi

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