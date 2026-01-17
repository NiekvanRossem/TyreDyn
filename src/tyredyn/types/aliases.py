from typing import Union, TypeAlias, Literal
from numpy import ndarray

# create alias for allowable input types
NumberLike : TypeAlias = int | float
SignalLike : TypeAlias = NumberLike | list[NumberLike] | ndarray
AngleUnit = Literal["deg", "rad"]
