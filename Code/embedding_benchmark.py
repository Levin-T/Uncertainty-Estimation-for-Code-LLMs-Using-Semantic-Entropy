import json
from openAI_emb import OpenAI_emb
import os
import csv

'''
Calculates the performance score of every embedding LLM
The performance store gets calculated by adding the results of every embedding calculation
Result calculation:
    same behavior = 1. Model predicts 0.8 (cosin similarty) -> + 0.8
    same behavior = 0. Model predicts 0.4 (cosin similarty) -> - 0.4
performance score = 
0 + 0.8 + (-0.4) = 0.4
0 + 0.7 + (-0.8) -> 0  
'''


def save_results_to_file(model_performance_score, model_name):
    results_folder = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(results_folder, exist_ok=True)
    filepath = os.path.join(results_folder, f"results_embedding_benchmark.csv")

    data = [f"Model= {model_name} ", f" performance results = {model_performance_score}"]

    # Open the CSV file in append mode
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)
        print(f"successfully saved results for model: {model_name}")


def runBenchmark(embedding_model, model_name):
    try:
        with open("functions_similarity_dataset.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: File not found at functions_similarity_dataset.json")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from file at functions_similarity_dataset.json")
        return

    performance_score = 0

    for index, item in enumerate(data):
        code_snipped_1 = item.get("function1")
        code_snipped_2 = item.get("function2")
        same_behavior = item.get("same_behavior")
        model_result = round(embedding_model.calculate_similarity(code_snipped_1, code_snipped_2), 2)
        if same_behavior == 0:
            performance_score = round(performance_score - model_result, 2)
        else:
            performance_score = round(performance_score + model_result, 2)
        print(f"finished iteration: {index}")
    save_results_to_file(performance_score, model_name)


if __name__ == '__main__':
    embedding_model_openAI = OpenAI_emb("text-embedding-3-large")
    runBenchmark(embedding_model_openAI, "text-embedding-3-large")
