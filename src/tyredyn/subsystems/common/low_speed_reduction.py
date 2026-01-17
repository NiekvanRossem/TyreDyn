import numpy as np
from tyredyn.types.aliases import SignalLike

class LowSpeedReduction:
    """Class containing the low speed correction equations."""

    def __init__(self, model):
        """Make the properties of the overarching class and other subsystems available."""
        self._model = model
        self._normalize = model.normalize

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _find_smooth_reduction(
            self,
            VX : SignalLike
    ) -> SignalLike:
        """Smooth low speed reduction factor."""

        # smooth reduction factor for low speed correction (MFeval)
        smooth_reduction = 1.0 - 0.5 * (1.0 + np.cos(np.pi * VX / self.VXLOW))
        return smooth_reduction
