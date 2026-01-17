from importlib.metadata import version

# Public API import
from tyredyn.models.magic_formula_61 import MF61
from tyredyn.models.magic_formula_62 import MF62
from tyredyn.io.initialize_tyre import Tyre

# current version of the package
__version__ = version("TyreDyn")

# control what gets imported when this package is loaded
__all__ = ["MF61", "MF62", "Tyre"]
