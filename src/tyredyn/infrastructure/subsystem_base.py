class SubSystemBase:
    """Base class for the subsystems of the tyre model."""

    def __init__(self, model):
        """Import the properties of the overarching class."""
        self._model = model

    def __getattr__(self, name):
        """Make the tyre coefficients directly available."""
        return getattr(self._model, name)

    def _connect(self, model):
        """Connect other subsystems. Default is no dependencies."""
        pass
