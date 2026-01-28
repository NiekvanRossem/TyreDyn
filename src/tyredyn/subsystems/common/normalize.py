from tyredyn.types.aliases import SignalLike, NumberLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase
from typing import Union, Literal
import numpy as np

class Normalize(SubSystemBase):
    """Module containing the functions to normalize input signals."""

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