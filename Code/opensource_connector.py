import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig  # For 8-bit quantization
from LLM import LLM
from openAI import get_test_prompt, get_system_prompt


class ConnectorClass(LLM):

    def __init__(self, model_name, quantize=False, load_model=True):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.quantize = quantize
        if load_model:
            self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.quantize:
            # Use 8-bit quantization to reduce GPU memory usage
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()

    def return_testcases(self, prompt, max_tokens, temp, number_of_test_cases):
        test_system_prompt = '''
        Keep input data within each test case CONCISE, especially list lengths (MAX_LENGTH = 5).
        Focus on VARIED test cases. Cover different scenarios (e.g., empty lists, single-element lists, lists with duplicates, different threshold values relative to element differences, boundary conditions).
        DO NOT generate long sequences of tests that only slightly modify one numerical value (like testing thresholds 0.1, 0.11, 0.12 against the same list) or just add one element repeatedly. Prioritize distinct structural variations.
        Generate ONLY the Python code for the generated_test_case() function as specified above.
        DO NOT ADD COMMENTS in the generated code.
        ONLY generate the test function. And *NOT* any solution code. 
        '''

        user_prompt_content = (
                f"Function Description:\n```python\n{prompt}\n```\n\n"
                f"Generate test cases based on the following instructions:\n"
                + get_test_prompt(number_of_test_cases)
        )
        # print(get_test_prompt(number_of_test_cases))
        messages = [
            {"role": "system", "content": test_system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True,
            top_p=0.85,
            use_cache=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def get_name(self):
        return self.model_name

    def compute_top_tokens(self, scores, k):
        top1_tokens = []
        top_k_tokens = []
        for score in scores:
            probs = torch.softmax(score[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, k)
            tokens = [self.tokenizer.decode(idx.item()) for idx in top_indices]

            probs_list = [float(prob) for prob in top_probs.cpu().detach().numpy()]
            tokens_with_probs = list(zip(probs_list, tokens))

            top1_tokens.append(tokens_with_probs[0])
            top_k_tokens.append(tokens_with_probs)

        return top1_tokens, top_k_tokens

    def return_output_general(self, input_text, top_k=5, max_new_tokens=50, temp=1, multi_inference=False):
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": input_text}
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except AttributeError:
            text = f"System: {get_system_prompt()}\nUser: {input_text}\nAssistant:"

        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=temp,
                top_k=top_k,
                do_sample=True,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs),
                use_cache=True
            )
        generated_ids = outputs.sequences[0]
        generated_tokens = generated_ids[len(inputs[0]):]
        scores = outputs.scores
        top1_tokens, top_tokens_per_step = self.compute_top_tokens(scores, top_k)
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        result = {
            "response": response,
            "probs": top1_tokens
        }

        if not multi_inference:
            result["top_k_tokens"] = top_tokens_per_step

        return result

    def return_output_multi_inference(self, user_prompt, max_token, temperature):
        return self.return_output_general(user_prompt, 50, max_token, temperature)

    def return_output(self, user_prompt, max_tokens, temperature, top_k):
        return self.return_output_general(user_prompt, top_k, max_tokens, temperature)


if __name__ == '__main__':
    # Set quantize=True to enable 8-bit quantization.
    llm = ConnectorClass("meta-llama/Meta-Llama-3.1-8B-Instruct")
    prompt = '''
        def exchange(lst1, lst2):
        """
        In this problem, you will implement a function that takes two lists of numbers,
        and determines whether it is possible to perform an exchange of elements
        between them to make lst1 a list of only even numbers.
        There is no limit on the number of exchanged elements between lst1 and lst2.
        If it is possible to exchange elements between the lst1 and lst2 to make
        all the elements of lst1 to be even, return "YES". Otherwise, return "NO".
        For example:
        exchange([1, 2, 3, 4], [1, 2, 3, 4]) => "YES"
        exchange([1, 2, 3, 4], [1, 5, 3, 4]) => "NO"
        It is assumed that the input lists will be non-empty.
        """

    Another error:'<' not supported between instances of 'int' and 'list'
        '''
    token_limit = 2048
    temp = 0.4
    start = time.time()
    output = llm.return_testcases(prompt, token_limit, temp, 20)
    end = time.time()
    print(f"took {end - start:.4f} seconds.")

    print(output)
