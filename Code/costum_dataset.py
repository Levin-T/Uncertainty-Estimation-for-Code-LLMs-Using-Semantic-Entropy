# Abstract class for every Dataset
from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def load_dataset_iterable(self):
        pass

    @abstractmethod
    def get_next_line(self):
        pass

    @abstractmethod
    def get_prompt(self, data):
        pass

    @abstractmethod
    def get_test(self, data):
        pass

    @abstractmethod
    def get_name(self):
        pass