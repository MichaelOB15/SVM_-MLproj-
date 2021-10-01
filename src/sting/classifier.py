from abc import ABC, abstractmethod
from typing import Sequence

from sting.data import AbstractDataSet


class Classifier(ABC):
    @abstractmethod
    def train(self, data: AbstractDataSet) -> None:
        pass

    @abstractmethod
    def predict(self, data: AbstractDataSet) -> Sequence[int]:
        pass
