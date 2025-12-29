from src.utils.formatting import SignalLike

class Normalize:
    """
    Module containing the functions to normalize input signals.
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
