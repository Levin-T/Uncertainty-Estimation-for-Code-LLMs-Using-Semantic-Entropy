from datasets import load_dataset
from costum_dataset import Dataset


class HumanEval_connector(Dataset):
    dataset = None
    iterator = None

    def __init__(self, name):
        self.name = name
        self.load_dataset_iterable()

    def load_dataset_iterable(self):
        self.dataset = load_dataset(self.name, split="test")
        self.iterator = iter(self.dataset)

    def get_next_line(self):
        assert self.iterator is not None, "Iterator of the dataset is not initialized"
        return next(self.iterator)
    
    def reset(self):
        self.iterator = iter(self.dataset)

    def get_dataset(self):
        # Ensure the dataset is loaded
        assert self.dataset is not None, "Dataset is not loaded. Call load_total_dataset first."
        return self.dataset

    def get_prompt(self, data):
        return data['prompt']

    def get_test(self, data):
        return data['test']

    def get_name(self):
        return self.name
