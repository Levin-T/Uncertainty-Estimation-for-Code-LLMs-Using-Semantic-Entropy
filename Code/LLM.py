# abstract class for LLMs
from abc import ABC, abstractmethod


class LLM(ABC):

    @abstractmethod
    def return_output(self, user_prompt, max_tokens, temperature, top_k):
        pass

    @abstractmethod
    def return_output_multi_inference(self, user_prompt, max_token, temperature):
        pass

    @abstractmethod
    def return_testcases(self, prompt, max_tokens, temperature, number_of_test_cases):
        pass

    @abstractmethod
    def get_name(self):
        pass
