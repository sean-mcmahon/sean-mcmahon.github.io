from typing import List
from typing import Optional
import random

Label = str
Predictions = List[Label]
Actuals = List[Label]


class GenerateRandomClassificationResults:
    def __init__(
        self,
        number_samples: int,
        label_names: List[str],
        random_seed: Optional[int] = None,
    ):
        self.number_samples = number_samples
        self.label_names = label_names
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def generate(self, probability: float = 0.85) -> [Predictions, Actuals]:
        actuals = [random.choice(self.label_names) for _ in range(self.number_samples)]
        predictions = []
        for actual in actuals:
            random_confidence = random.random()
            if random_confidence < probability:
                prediction = actual
            else:
                prediction = random.choice(self.label_names)
            predictions.append(prediction)
        return predictions, actuals
