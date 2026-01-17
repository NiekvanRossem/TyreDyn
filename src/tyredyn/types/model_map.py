"""Links the model type to the subclass that needs to be created."""

from tyredyn import MF61, MF62

MODEL_CLASS_MAP = {
    "MF6.1": MF61,
    "MF6.2": MF62
}