import re
from datasets import load_dataset
from costum_dataset import Dataset


def extract_function_name_from_code(code):
    pattern = r"def\s+(\w+)\s*\(([^)]*)\)\s*(->\s*[^:]+)?\s*:"
    match = re.search(pattern, code)
    if match:
        func_name = match.group(1)
        params = match.group(2)
        ret_annotation = match.group(3) or ""
        return f"def {func_name}({params}){ret_annotation}"
    else:
        return "No function definition found."


class Mbpp_connector(Dataset):
    dataset = None
    iterator = None

    def __init__(self, name):
        self.name = name
        self.load_dataset_iterable()

    def load_dataset_iterable(self):
        self.dataset = load_dataset(self.name, "sanitized", split="test")
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
        return (f"{data['prompt']}\n"
                f"Follow the following function name and signature: "
                f"{extract_function_name_from_code(data['code'])}")

    def get_test(self, data):
        return data['test_list']

    def get_test_import(self, data):
        return data["test_imports"]

    def get_name(self):
        return self.name
