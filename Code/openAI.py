from helpers import get_api_key
import math
from LLM import LLM
from openai import OpenAI

system_prompt_text = '''
             You are an expert programming assistant. When given a function signature and description, "
             you must:"
              -Write only the function definition, including any required imports. And do not write anything else.
              -Ensure the function works with the provided example inputs and matches the expected output.
              -Do not include explanations, comments, or any additional text outside the code.
              -Provide the code as a raw string with no formatting or additional text.
              -Make sure that every import is included such as for example: from typing import List.
            '''
test_prompt_text = '''
        You are an expert programming assistant. When given a function prompt, you must:

        1. Identify relevant partitions of the input space based on the function description, including:
        2. From these partitions, generate exactly {N} diverse test cases in total that cover a wide range of scenarios. Each test case should be a tuple of (args, expected), where args is a tuple of input arguments to the function, and expected is the correct output as per the function description.

        **Output Format:**
        Generate *only* a Python function `generated_test_case()` structured exactly like this:
        ```python
        def generated_test_case():
            test_cases = [
                # ... exactly {N} tests cases
            ]
            # Add a loop here to run the test cases against the function
            # being tested (assume it exists elsewhere) and count/return passed tests.
            passed = 0
            # Example loop structure (adapt based on function signature):
            # for args_tuple, expected in test_cases:
            #     # Assuming function_under_test takes args from the tuple
            #     result = function_under_test(*args_tuple) 
            #     if result == expected:
            #         passed += 1
            return passed
            '''


def get_system_prompt():
    return system_prompt_text


def get_test_prompt(number_of_test_cases):
    return test_prompt_text.format(N=number_of_test_cases)


class ChatGpt(LLM):

    def __init__(self, model):
        # Fetch the API key using helper method
        self.client = OpenAI(api_key=get_api_key())
        self.model = model
        # Define the system prompt

        self.SYSTEM_PROMPT = (
            system_prompt_text
        )

    def return_testcases(self, p, max_tokens, temperature, number_of_test_cases):
        test_system_prompt = (
                "generate exactly " + str(number_of_test_cases) + "test cases\n" + test_prompt_text
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": test_system_prompt},
                {"role": "user", "content": f"The input function looks the following way: {p}"}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content

    def return_output(self, user_prompt, max_tokens, temperature, top_k=5):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_k,
        )
        probabilities = []
        top_k_tokens = []
        logprobs = completion.choices[0].logprobs

        for index, token_logprob in enumerate(logprobs.content):
            if index < 3 or index == len(logprobs.content) - 1 or index == len(logprobs.content) - 2:
                continue
            temp = []
            for top_logprob in token_logprob.top_logprobs:
                temp.append((math.exp(top_logprob.logprob), top_logprob.token))

            top_k_tokens.append(temp)
            probabilities.append((math.exp(token_logprob.logprob), token_logprob.token))

        return {
            "response": completion.choices[0].message.content,
            "probs": probabilities,
            "top_k_tokens": top_k_tokens
        }

    def return_output_multi_inference(self, user_prompt, max_tokens, temp):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            max_tokens=max_tokens,
            logprobs=True,
        )
        probabilities = []
        logprobs = completion.choices[0].logprobs

        for index, token_logprob in enumerate(logprobs.content):
            if index < 3 or index == len(logprobs.content) - 1 or index == len(logprobs.content) - 2:
                continue
            probabilities.append((math.exp(token_logprob.logprob), token_logprob.token))

        return {
            "response": completion.choices[0].message.content,
            "probs": probabilities,
        }

    def get_name(self):
        return self.model