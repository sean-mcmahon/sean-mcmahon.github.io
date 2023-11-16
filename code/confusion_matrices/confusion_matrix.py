from typing import Annotated
from typing import Literal
from dataclasses import dataclass

from numpy import int_
from numpy.typing import NDArray


@dataclass
class ConfusionMatrix:
    matrix_array: Annotated[NDArray[int_], Literal["n_classes", "n_classes"]]
    labels: tuple[str, ...]
