"""Links the model type to the subclass that needs to be created."""

from src.tyre_models.magic_formula_61 import MF61
from src.tyre_models.magic_formula_62 import MF62

MODEL_CLASS_MAP = {
    "MF6.1": MF61,
    "MF6.2": MF62
}