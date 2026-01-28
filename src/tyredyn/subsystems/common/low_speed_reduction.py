import numpy as np
from tyredyn.types.aliases import SignalLike
from tyredyn.infrastructure.subsystem_base import SubSystemBase

class LowSpeedReduction(SubSystemBase):
    """Class containing the low speed correction equations."""

    def _connect(self, model):
        self._normalize = model.normalize

    def _smooth_reduction(
            self,
            VX : SignalLike
    ) -> SignalLike:
        """Smooth low speed reduction factor."""

        # smooth reduction factor for low speed correction (MFeval)
        smooth_reduction = 1.0 - 0.5 * (1.0 + np.cos(np.pi * VX / self.VXLOW))
        return smooth_reduction
