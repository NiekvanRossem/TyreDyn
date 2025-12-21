"""Links the model type to the subclass that needs to be created."""

#from src.models.magic_formula_61 import MF61
from src.base.magic_formula_61_new import MF61new

MODEL_CLASS_MAP = {
    "MF6.1": MF61new,
    "MF6.2": MF61new # TODO: fix this after implementing MF 6.2
}