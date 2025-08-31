# benchmark framework
from helpers import *
from LLM import LLM
import logging
from uncertainty_calculations import *
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import json
from mbpp import Mbpp_connector
from helpers import execute_generated_code_mbpp
from humanEval import HumanEval_connector
from openAI import ChatGpt
from opensource_emb import OpenSourceEmb
import json
import base64
import zlib
import textwrap

from opensource_connector import ConnectorClass

logger = logging.getLogger(__name__)

def parse_params(line):
    """
    Parses a parameters line such as:
    "temp: 0.2---amount iterations: 2---bleu threshhold: 0.0---emb threshold: 0.9"
    Returns a dictionary with keys: temp, amount iterations, bleu threshhold, emb threshold.
    """
    params = {}
    parts = line.split('---')
    for part in parts:
        key, val = part.split(':', 1)
        params[key.strip()] = val.strip()
    return params

def parse_scores(line):
    """
    Parses the scores line, converting each comma-separated value to float.
    Returns a list of floats.
    """
    return [float(x) for x in line.split(',') if x.strip()]

import datetime

def analyze_result_file(file_path, amount_of_tests, output_path="results.txt"):
    records = []
    # Read input, skipping blank lines
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    result_dic = {}
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        params = parse_params(lines[i])
        scores = parse_scores(lines[i+1])

        embedding_max = max(scores[0:2])
        code_bleu_max = max(scores[2:4])
        testing_max   = max(scores[4:6])
        combined_max  = max(scores[6:8])

        metrics = [
            ("emb",  embedding_max),
            ("bleu", code_bleu_max),
            ("test", testing_max),
            ("comb", combined_max),
        ]

        temp_key = f"temp-{params['temp']}"
        entry = result_dic.setdefault(temp_key, {
            "emb":0, "bleu":0, "test":0, "comb":0,
            "emb_called":None, "bleu_called":None,
            "test_called":None, "comb_called":None,
        })

        for name, val in metrics:
            if val > entry[name]:
                entry[name] = val
                entry[f"{name}_called"] = params

    # helper to safely get a param
    def gp(d, k): return d.get(k, "N/A") if d else "N/A"

    # append to the output file
    with open(output_path, "a") as out:
        # header for this run
        out.write(f"===={amount_of_tests} Test cases ====\n\n")

        for temp, data in result_dic.items():
            out.write(f"{temp}\n\n")
            '''
            out.write(f"Embedding: {data['emb']:.9f}\n")
            e = data['emb_called']
            out.write(f"Emb threshold: {gp(e,'emb threshold')} | iterations: {gp(e,'amount iterations')}\n")
            out.write("-" * 10 + "\n\n")

            out.write(f"Code BLEU: {data['bleu']:.9f}\n")
            b = data['bleu_called']
            out.write(f"Bleu threshhold: {gp(b,'bleu threshhold')} | iterations: {gp(b,'amount iterations')}\n")
            out.write("-" * 10 + "\n\n")
            '''
            out.write(f"Test: {data['test']:.9f}\n")
            t = data['test_called']
            out.write(f"Iterations: {gp(t,'amount iterations')}\n")
            out.write("-" * 10 + "\n\n")

            out.write(f"Combined: {data['comb']:.9f}\n")
            c = data['comb_called']
            out.write(
                f"Iterations: {gp(c,'amount iterations')} | "
                f"Emb threshold: {gp(c,'emb threshold')} | "
                f"Bleu threshhold: {gp(c,'bleu threshhold')}\n"
            )
            out.write("-" * 30 + "\n\n")

def extract_code(text: str) -> str:
    text = textwrap.dedent(text)
    lines = text.splitlines()
    valid = ("def ", "from ", "import ", "class ", "@")
    while lines and (not lines[0].strip() or not lines[0].lstrip().startswith(valid)):
        lines.pop(0)
    while lines:
        candidate = "\n".join(lines)
        try:
            compile(candidate, "<string>", "exec")
            return candidate
        except SyntaxError:
            lines.pop()
    return ""

def save_results_to_file(model, results, dataset, single_inference=False,
                         inference_count=None, bleu_threshold=None, emb_threshold=None, temperature=None):
    # Added optional 'temperature' parameter and updated header.
    results_folder = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(results_folder, exist_ok=True)
    if single_inference:
        csv_filepath = os.path.join(results_folder, f"results_single_inference_{model}_{dataset}.csv")
    else:
        csv_filepath = os.path.join(results_folder, f"results_multi_inference_{model}_{dataset}.csv")

    results_df = pd.DataFrame([results])
    result_string = results_df.to_csv(header=False, index=False)
    
    header_line = ""
    # Build header if all parameters are provided (including temperature)
    if temperature is not None and inference_count is not None and bleu_threshold is not None and emb_threshold is not None:
        header_line = f"temp: {temperature}---amount iterations: {inference_count}---bleu threshhold: {bleu_threshold}---emb threshold: {emb_threshold}\n"

    if os.path.exists(csv_filepath):
        with open(csv_filepath, "a") as f:
            if header_line:
                f.write(header_line)
            f.write(result_string)
    else:
        with open(csv_filepath, "w") as f:
            if header_line:
                f.write(header_line)
            f.write(result_string)

    #print(f"Saved results for: {model} to {csv_filepath}")


def filter_common_tokens(token_list):
    """
    Filters out unnecessary tokens (common boilerplate) from a list of (prob, token) tuples.

    :param token_list: List of tuples (float, str)
    :return: Filtered list of tuples (float, str) without common boilerplate tokens.
    """
    # Define tokens you consider "common" or "unnecessary"
    UNNECESSARY_TOKENS = {
        "from", "import", "typing", "List", "def", "return",
        # Add or remove tokens as needed:
        # for instance, you might also exclude "class", "if __name__ == '__main__':" etc.
    }

    filtered_tokens = []

    for prob, token in token_list:
        # Strip leading/trailing whitespace to match the tokens properly
        stripped_token = token.strip()

        # Check if it's in the set of unnecessary tokens
        if stripped_token in UNNECESSARY_TOKENS:
            continue

        # You may also want to filter out pure newlines or pure spaces:
        if not stripped_token:  # skip if empty after strip
            continue

        # Otherwise, keep the token
        filtered_tokens.append((prob, token))

    return filtered_tokens


def save_arrays_single_inference(filename, avg_props, max_props, avg_entropies, max_entropies, results_separated):
    np.savez_compressed(
        filename,
        avg_props=avg_props,
        max_props=max_props,
        avg_entropies=avg_entropies,
        max_entropies=max_entropies,
        results_separated=results_separated
    )
    print(f"Arrays stored temporarily to {filename}.")


def save_arrays_multi_inference(
        filename,
        emb_multi_inference_no_ln,
        emb_multi_inference_with_ln,
        syn_multi_inference_no_ln,
        syn_multi_inference_with_ln,
        tests_multi_inference_no_ln,
        tests_multi_inference_with_ln):
    np.savez_compressed(
        filename,
        emb_multi_inference_no_ln=emb_multi_inference_no_ln,
        emb_multi_inference_with_ln=emb_multi_inference_with_ln,
        syn_multi_inference_no_ln=syn_multi_inference_no_ln,
        syn_multi_inference_with_ln=syn_multi_inference_with_ln,
        tests_multi_inference_no_ln=tests_multi_inference_no_ln,
        tests_multi_inference_with_ln=tests_multi_inference_with_ln
    )
    print(f"Multi-inference arrays saved to {filename}.")


def save_results_single_inference(avg_props, max_props, avg_entropies, max_entropies, results,
                                  model_name, dataset_name):
    # In this case every observed instance is either wrong or true (almost 0% chance)
    # However ensures that calculated values are not lost.
    if len(np.unique(results)) < 2:
        print("Error! Not enough unique values to perform single inference aoc calculations.\n"
              "Persist results without calculated scores")
        save_arrays_single_inference(
            filename=f"persisted_results_single_inference_{str(model_name)}_{str(dataset_name)}",
            avg_props=avg_props,
            max_props=max_props,
            avg_entropies=avg_entropies,
            max_entropies=max_entropies,
            results_separated=results)
    else:
        auc_scores_single_inference = {
            "auc_avg_props": roc_auc_score(results, 1 - np.array(avg_props)),
            "auc_max_props": roc_auc_score(results, 1 - np.array(max_props)),
            "auc_avg_entropies": roc_auc_score(results, 1 - np.array(avg_entropies)),
            "auc_avg_max_entropies": roc_auc_score(results, 1 - np.array(max_entropies)),
        }

        save_results_to_file(
            model_name,
            auc_scores_single_inference,
            dataset_name,
            True)


def save_results_multi_inference(
        passed_at_k,
        emb_no_ln,
        emb_with_ln,
        syn_no_ln,
        syn_with_ln,
        tests_no_ln,
        test_with_ln,
        combined,
        combined_with_ln,
        model_name,
        dataset_name,
        inference_count=None,         # <-- Extra header information
        bleu_threshold=None,          # (code threshold corresponds to BLEU_THRESHOLD)
        emb_threshold=None,           # <-- Extra header information
        temperature=None             # <-- New temperature parameter added
):
    auc_scores_multi_inference = {
        "auc_emb_multi_inference_no_ln": 1 - roc_auc_score(passed_at_k, np.array(emb_no_ln)),
        "auc_emb_multi_inference_with_ln": 1 - roc_auc_score(passed_at_k, np.array(emb_with_ln)),
        "auc_syn_multi_inference_no_ln": 1 - roc_auc_score(passed_at_k, np.array(syn_no_ln)),
        "auc_syn_multi_inference_with_ln": 1 - roc_auc_score(passed_at_k, np.array(syn_with_ln)),
        "auc_test_based_multi_inference": 1 - roc_auc_score(passed_at_k, np.array(tests_no_ln)),
        "auc_test_based_with_ln_multi_inference": 1 - roc_auc_score(passed_at_k, np.array(test_with_ln)),
        "combined_multi_inference_no_ln": 1 - roc_auc_score(passed_at_k, np.array(combined)),
        "combined_multi_inference_with_ln": 1 - roc_auc_score(passed_at_k, np.array(combined_with_ln)),
    }
    save_results_to_file(model_name, auc_scores_multi_inference, dataset_name,
                         False, inference_count, bleu_threshold, emb_threshold, temperature)


def remove_tokens_with_prob_one(token_probs):
    return [pair for pair in token_probs if pair[0] != 1.0]


def format_json_token_props(list_of_list):
    result_list = []
    for input_list in list_of_list:
        result_list.append((input_list[0], input_list[1]))
    return result_list


def format_json_top_k_props(input_list):
    result_list = []
    for tmp in input_list:
        tmp_list = []
        for token_tuple_list in tmp:
            tmp_list.append((token_tuple_list[0], token_tuple_list[1]))
        result_list.append(tmp_list)
    return result_list


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def compress_vector(emb_vector):
    return base64.b64encode(
        zlib.compress(emb_vector.tobytes())
    ).decode('utf-8')


def decompress_vector(compressed_vector):
    return np.frombuffer(
        zlib.decompress(base64.b64decode(compressed_vector)),
        dtype=np.float32
    )


class Benchmark:

    def __init__(self, dataset, llm, embedding_model):
        self.dataset = dataset
        self.llm = llm
        self.embedding_model = embedding_model
        model_name = self.llm.get_name().replace("/", "_")
        # We use base file paths and later add temperature info if needed.
        self.base_filepath_multi = f"Inference/multi_inference_results_{model_name}_{self.dataset.get_name()}"
        self.base_filepath_single = f"Inference/single_inference_results_{model_name}_{self.dataset.get_name()}"

        # make sure that the data set is correctly initialized
        assert isinstance(self.llm, LLM)

        logging.basicConfig(filename='logging.log', level=logging.INFO)

    def get_filepath_multi_inference(self, temp=None):
        if temp is None:
            return f"{self.base_filepath_multi}.json"
        else:
            # Format temperature to two decimals and remove the decimal point
            temp_str = f"{temp:.2f}".replace(".", "")
            return f"{self.base_filepath_multi}_{temp_str}.json"

    def get_filepath_single_inference(self, temp=None):
        if temp is None:
            return f"{self.base_filepath_single}.json"
        else:
            temp_str = f"{temp:.2f}".replace(".", "")
            return f"{self.base_filepath_single}_{temp_str}.json"

    def create_json_single_inference(self, token_limit, top_k_tokens, tmp):
        logger.info("-------->Starting model single inference<--------\n")
        result_list = []
        current_problem = 0
        filepath = self.get_filepath_single_inference(tmp)
        while True:
            try:
                current_line = self.dataset.get_next_line()
                prompt_dataset = self.dataset.get_prompt(current_line)

                logger.info(f"Currently in iteration: {str(current_problem)}\n")
                logger.info(f"Human_eval_test cases: \n{str(self.dataset.get_test(current_line))}\n")

                result_dictionary_instance = {
                    "dataset_test": self.dataset.get_test(current_line)
                }
                if isinstance(dataset, Mbpp_connector):
                    result_dictionary_instance["test_import_list"] = self.dataset.get_test_import(current_line)

                output = self.llm.return_output(
                    user_prompt=prompt_dataset,
                    max_tokens=token_limit,
                    temperature=tmp,
                    top_k=top_k_tokens)

                generated_code = (str(output["response"])
                                  .replace("```", "")
                                  .replace("python", "")
                                  .replace("\n", "", 1))

                logger.info("Code for the current_Problem: \n" + generated_code + "\n")
                result_dictionary_instance["generated_code"] = generated_code
                result_dictionary_instance["token_probs"] = json.dumps(output["probs"], separators=(',', ':'))
                result_dictionary_instance["top_k_tokens"] = json.dumps(output["top_k_tokens"], separators=(',', ':'))

                print(f"done with problem {current_problem}")
                result_list.append(result_dictionary_instance)
                current_problem += 1

            except StopIteration:
                self.dataset.reset()
                try:
                    with open(filepath, "w") as file:
                        json.dump(result_list, file, indent=1)
                except Exception as e:
                    print(f"Error writing to json file: {e}", e)
                break

    def create_json_multi_inference(self, token_limit, temp, number_of_inferences, number_of_test_cases):
        logger.info("-------->Starting model multi inference<--------\n")
        current_problem = 0

        # Prepare the output file. Opening with "w" once at the start clears old content:
        filepath = self.get_filepath_multi_inference(temp)
        with open(filepath, "w") as f:
            f.write("[\n")

        first_object = True

        while True:
            try:
                # 1. Get the next line / prompt
                current_line = self.dataset.get_next_line()
                prompt = self.dataset.get_prompt(current_line)

                # 2. Generate test cases
                generated_testcase = self.llm.return_testcases(
                    prompt,
                    10000,
                    0.4,
                    number_of_test_cases
                )
                generated_testcase = (
                    str(generated_testcase)
                    .replace("```", "")
                    .replace("python", "")
                )

                logger.info(f"Currently in iteration: {current_problem}")
                logger.info(f"Generated test case:\n{generated_testcase}")
                logger.info(f"Dataset test cases:\n{self.dataset.get_test(current_line)}")

                # 3. Build a dictionary for this problem
                tmp_dictionary = {
                    "problem_inferences": [],
                    "generated_test": generated_testcase,
                    "dataset_test": self.dataset.get_test(current_line)
                }
                if isinstance(self.dataset, Mbpp_connector):
                    tmp_dictionary["test_import_list"] = self.dataset.get_test_import(current_line)

                # 4. Generate multiple inferences
                for i in range(number_of_inferences):
                    output = self.llm.return_output_multi_inference(prompt, token_limit, temp)
                    generated_code = (
                        str(output["response"])
                        .replace("```", "")
                        .replace("python", "")
                        .replace("\n", "", 1)
                    )
                    logger.info(f"Code for inference {i}:\n{generated_code}")

                    inference_result = {
                        "inference_iteration": i,
                        "generated_code": generated_code,
                        # Store token probs as compact JSON string
                        "token_probs": json.dumps(output["probs"], separators=(',', ':')),
                        # Omit or clear embedding if it's too large
                        "embedding": ""
                    }
                    tmp_dictionary["problem_inferences"].append(inference_result)

                obj_json = json.dumps(tmp_dictionary, indent=1)

                with open(filepath, "a") as f:
                    if not first_object:
                        f.write(",\n")
                    else:
                        first_object = False

                    f.write(obj_json)
                print(f"done with {current_problem}")
                current_problem += 1
            except StopIteration:
                dataset.reset()
                # No more problems
                break

        with open(filepath, "a") as f:
            f.write("\n]\n")
        logger.info("-------->All problems processed<--------")

    def load_and_calculate_single_inference(self, temp):
        filepath = self.get_filepath_single_inference(temp)

        try:
            with open(filepath, "r") as file:
                data_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file at {filepath}")
            return

        avg_props = []
        max_props = []
        avg_entropies = []
        max_entropies = []
        model_name = self.llm.get_name().replace("/", "_")
        results = read_json(f"Inference/multi_inference_performance_results_{model_name}_"
                            f"{self.dataset.get_name()}.json")
        for index, item in enumerate(data_json):
            token_probs = format_json_token_props(json.loads(item["token_probs"]))
            top_k_tokens = format_json_top_k_props(json.loads(item["top_k_tokens"]))

            avg_props.append(
                calculate_avg_prob_single_inference(token_probs)
            )
            max_props.append(
                calculate_max_prob_single_inference(token_probs)
            )
            avg_entropies.append(
                calculate_avg_entropy_single_inference(top_k_tokens)
            )
            max_entropies.append(
                calculate_max_entropy_single_inference(top_k_tokens)
            )
        save_results_single_inference(avg_props, max_props, avg_entropies, max_entropies, results,
                                      model_name, self.dataset.get_name())

    def load_and_calculate_multi_inference(self, BLEU_THRESHOLD=0.0, EMB_THRESHOLD=0.0, amount_inferences=5, temp=None):
        # Use the temperature to determine which file to load.
        filepath = self.get_filepath_multi_inference(temp)
        logger.info("-------->Starting calculations for multi-inference<--------\n")
        try:
            with open(filepath, "r") as file:
                json_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file at {filepath}")
            return
        model_name = self.llm.get_name().replace("/", "_")
        temp_str = f"{temp:.2f}".replace(".", "")

        generated_tests_result = read_json(
            f"Performance/generated_tests_performance_results_"
            f"{model_name}_{self.dataset.get_name()}_{temp_str}.json"
        )
        emb_no_ln = []
        emb_with_ln = []
        syn_no_ln = []
        syn_with_ln = []
        tests_no_ln = []
        tests_with_ln = []
        combined = []
        combined_with_ln = []
        counter = 0
        for i, problem_data in enumerate(json_data):
            result_generated_tests_tmp = []
            result_dataset_tests_per_problem = []
            equiv_list = []
            for j, inference in enumerate(problem_data["problem_inferences"]):
                if j < amount_inferences:
                    generated_code = inference["generated_code"]
                    token_props = format_json_token_props(json.loads(inference["token_probs"]))
                    embeddings = decompress_vector(inference["embedding"])
                    equiv_list.append(
                        calculate_snipped_probabilities_and_length(generated_code,
                                                                   token_props,
                                                                   generated_tests_result[i][j],
                                                                   embeddings)
                    )
            logger.info("Passed the following Generated tests: " + str(result_generated_tests_tmp) + "\n")
            logger.info("Performance on Dataset testcases  " + str(result_dataset_tests_per_problem) + "\n")

            equiv_calculation_methods = [Similarity_Calculations.SYNTACTIC,
                                         Similarity_Calculations.TEST_BASED,
                                         Similarity_Calculations.EMBEDDING,
                                         Similarity_Calculations.COMBINED
                                         ]

            for calculation_method in equiv_calculation_methods:
                equiv_classes = cluster(equiv_list,
                                        BLEU_THRESHOLD,
                                        EMB_THRESHOLD,
                                        calculation_method)
                equiv_entropy = calculate_entropy_for_equiv_classes(equiv_classes)
                equiv_entropy_length_normalized = calculate_entropy_for_equiv_classes(equiv_classes, True)
                logger.info(
                    f"Calculated: EVE(x) = {equiv_entropy} | EVE(X)_length_normalized = "
                    f"{equiv_entropy_length_normalized} using: {calculation_method}")

                if calculation_method == Similarity_Calculations.SYNTACTIC:
                    syn_no_ln.append(equiv_entropy)
                    syn_with_ln.append(equiv_entropy_length_normalized)
                elif calculation_method == Similarity_Calculations.EMBEDDING:
                    emb_no_ln.append(equiv_entropy)
                    emb_with_ln.append(equiv_entropy_length_normalized)
                elif calculation_method == Similarity_Calculations.TEST_BASED:
                    tests_no_ln.append(equiv_entropy)
                    tests_with_ln.append(equiv_entropy_length_normalized)
                else:
                    combined.append(equiv_entropy)
                    combined_with_ln.append(equiv_entropy_length_normalized)

            counter += 1
        save_results_multi_inference(
            read_json(f"Inference/multi_inference_performance_results_{model_name}_{self.dataset.get_name()}.json"),
            emb_no_ln,
            emb_with_ln,
            syn_no_ln,
            syn_with_ln,
            tests_no_ln,
            tests_with_ln,
            combined,
            combined_with_ln,
            model_name,
            self.dataset.get_name(),
            amount_inferences,      # amount iterations
            BLEU_THRESHOLD,         # bleu/code threshold
            EMB_THRESHOLD,          # embedding threshold
            temp                    # temperature
        )

    def compute_llm_performance(self):
        logger.info("-------->Compute LLM performance<--------\n")
        model_name = self.llm.get_name().replace("/", "_")
        result_list = []
        current_problem = 0
        while True:
            try:
                current_line = self.dataset.get_next_line()
                prompt_dataset = self.dataset.get_prompt(current_line)

                dataset_test = self.dataset.get_test(current_line)

                output = self.llm.return_output(
                    user_prompt=prompt_dataset,
                    max_tokens=500,
                    temperature=0.01,
                    top_k=1)

                generated_code = (str(output["response"])
                                  .replace("```", "")
                                  .replace("python", "")
                                  .replace("\n", "", 1))
                logger.info("Code for the current_Problem: \n" + generated_code + "\n")

                if isinstance(self.dataset, Mbpp_connector):
                    result = execute_generated_code_mbpp(generated_code,
                                                         self.dataset.get_test_import(current_line),
                                                         dataset_test)
                else:
                    result = execute_generated_code_normal(generated_code,
                                                           dataset_test)
                result_list.append(result)
                print(f"result: {result}")
                print(f"done with problem {current_problem}")
                current_problem += 1

            except StopIteration:
                try:
                    with open(f"multi_inference_performance_results_{model_name}_{self.dataset.get_name()}.json",
                              "w") as file:
                        json.dump(result_list, file, indent=1)
                except Exception as e:
                    logger.info(f"Error writing to json file: {e}", e)
                break

    def run_generated_tests(self, temp=None):
        logger.info("-------->Compute generated test performance<--------\n")
        result_list = []
        current_problem = 0
        model_name = self.llm.get_name().replace("/", "_")

        # Read from the multi-inference file that includes temperature info (if provided)
        with open(self.get_filepath_multi_inference(temp), "r") as file:
            json_data = json.load(file)

        for problem_data in json_data:
            print(f"starting iteration {current_problem}")
            tmp_result_list = []
            generated_testcase = extract_code(problem_data["generated_test"])
            for inference in problem_data["problem_inferences"]:
                generated_code = extract_code(inference["generated_code"])
                tmp_result_list.append(
                    execute_generated_tests(generated_code, generated_testcase)
                )
            logger.info("Passed the following Generated tests: " + str(current_problem) + "\n")
            logger.info("Performance on Dataset testcases  " + str(tmp_result_list) + "\n")

            result_list.append(tmp_result_list)
            current_problem += 1
        if temp is None:
            out_filepath = f"Performance/generated_tests_performance_results_{model_name}_{self.dataset.get_name()}.json"
        else:
            temp_str = f"{temp:.2f}".replace(".", "")
            out_filepath = f"Performance/generated_tests_performance_results_{model_name}_{self.dataset.get_name()}_{temp_str}.json"

        with open(out_filepath, "w") as file:
            json.dump(result_list, file, indent=1)

    def generate_embedding(self, tmp):
        filepath = self.get_filepath_multi_inference(tmp)
        result_list = []
        with open(filepath, "r") as file:
            json_data = json.load(file)
        for iteration, problem_data in enumerate(json_data):
            for inference in problem_data["problem_inferences"]:
                emb_vector = self.embedding_model.get_embedding(inference["generated_code"])
                inference["embedding"] =  compress_vector(np.array(emb_vector, dtype=np.float32))
            result_list.append(problem_data)
            print(f"done with embeddings for problem {iteration}")
        with open(filepath, "w") as file:
            json.dump(result_list, file, indent=1)

    def generate_tests(self, tmp_generated_file, amount_of_tests):
        filepath = self.get_filepath_multi_inference(tmp_generated_file)
        result_list = []
        with open(filepath, "r") as file:
            json_data = json.load(file)
            for counter, problem_data in enumerate(json_data):
                current_problem = self.dataset.get_next_line()
                problem_data["generated_test"] = self.llm.return_testcases(
                    current_problem,
                    10000,
                    0.4,
                    amount_of_tests
                )
                result_list.append(problem_data)
                print(f"done with problem {counter}")
        self.dataset.reset()
        with open(filepath, "w") as file:
            json.dump(result_list, file, indent=1)
            print(f"wrote to file: {filepath}")
    '''
    Count the amount of test generations that were successful.
    '''
    def count_tests_generated(self, temp):
        model_name = self.llm.get_name().replace("/", "_")
        temp_str = f"{temp:.2f}".replace(".", "")
        testcases = 0
        amount_testcases_per_inference = 0
        generated_tests_result = read_json(
            f"Performance/generated_tests_performance_results_"
            f"{model_name}_{self.dataset.get_name()}_{temp_str}.json"
        )
        for testcase in generated_tests_result: 
            for iteration in testcase: 
                if iteration == -1 or iteration == 404: 
                    amount_testcases_per_inference += 1
            if amount_testcases_per_inference < 30:
                testcases += 1
            amount_testcases_per_inference = 0
        return testcases
    
    def reset_files(self):
        files = [
            "/home/levin/work_levin/thesis/Code/Performance/generated_tests_performance_results_gpt-4.1-mini_mbpp_020.json",
            "/home/levin/work_levin/thesis/Code/Performance/generated_tests_performance_results_gpt-4.1-mini_mbpp_050.json",
            "/home/levin/work_levin/thesis/Code/Performance/generated_tests_performance_results_gpt-4.1-mini_mbpp_070.json",
            "/home/levin/work_levin/thesis/Code/Results/results_multi_inference_gpt-4.1-mini_mbpp.csv"
        ]
        for fn in files:
            try:
                os.remove(fn)
                # print(f"Deleted {fn}")  # optional logging
            except FileNotFoundError:
                # file wasn’t there, ignore
                print("couldnt find file")
            except OSError as e:
                # something else went wrong (permissions?), warn if you like
                print(f"Could not delete {fn}: {e}")

if __name__ == '__main__':
    #dataset_string = "mbpp"
    dataset_string = "openai_humaneval"
    file_name = f"results_multi_inference_gpt-4.1-mini_{dataset_string}.csv"
    model = "gpt-4.1-mini"
    emb_model = OpenSourceEmb('Salesforce/SFR-Embedding-Code-2B_R', 32768, False)
    dataset = HumanEval_connector("openai_humaneval")
    #dataset = Mbpp_connector("mbpp")
    amount_of_tests = 30
    while amount_of_tests <= 50:
        llm = ChatGpt("gpt-4.1-mini")
        benchmark = Benchmark(dataset=dataset, llm=llm, embedding_model="")
        benchmark.reset_files()
        benchmark.generate_tests(tmp_generated_file=0.2, amount_of_tests=amount_of_tests)
        benchmark.generate_tests(tmp_generated_file=0.5, amount_of_tests=amount_of_tests)
        benchmark.generate_tests(tmp_generated_file=0.7, amount_of_tests=amount_of_tests)

        benchmark.run_generated_tests(0.2)
        benchmark.run_generated_tests(0.5)
        benchmark.run_generated_tests(0.7)

        for i in range(1, 31):
            emb_threshold = 0.9
            code_threshold = 0.0
            for _ in range(0, 10):
                benchmark.load_and_calculate_multi_inference(code_threshold, emb_threshold, i, 0.2)
                benchmark.load_and_calculate_multi_inference(code_threshold, emb_threshold, i, 0.5)
                benchmark.load_and_calculate_multi_inference(code_threshold, emb_threshold, i, 0.7)
                emb_threshold += 0.01
                code_threshold += 0.1
                #print(f"done with {model} [inference: {i}] and emb: {emb_threshold} + code: {code_threshold}")
        print(f"done with model {model}")
        result_line = (
            f"Generated tests: {amount_of_tests}\n"
            f"{benchmark.count_tests_generated(0.2)}, "
            f"{benchmark.count_tests_generated(0.5)}, "
            f"{benchmark.count_tests_generated(0.7)}\n"
        )
        with open("generated_tests.txt", "a") as out:
            out.write(result_line)

        analyze_result_file(f"Results/{file_name}", amount_of_tests)
        print(f"done with {amount_of_tests} test cases")
        amount_of_tests += 5
