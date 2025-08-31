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
def analayze_result_file(file_path):
    records = []
    # Read the file data.csv. Adjust the filename as needed.
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # skip empty lines

    
    result_dic = {}
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break  
        param_line = lines[i]
        score_line = lines[i + 1]
        
        params = parse_params(param_line)
        scores = parse_scores(score_line)

        embedding_max = max(scores[0:2])
        code_bleu_max = max(scores[2:4])
        testing_max = max(scores[4:6])
        combined_max = max(scores[6:8])

        metrics = [
            ("emb",  embedding_max),
            ("bleu", code_bleu_max),
            ("test", testing_max),
            ("comb", combined_max),
        ]
        temp = f"temp-{params["temp"]}"
        data_for_temp = result_dic.setdefault(temp, {
            "emb": 0, "bleu": 0, "test": 0, "comb": 0,
            "emb_called": 0, "bleu_called": 0, "test_called": 0, "comb_called": 0
        })

        # Step 3: Loop through our metrics and update the dictionary if the new value is larger
        for metric_name, metric_value in metrics:
            if metric_value > data_for_temp[metric_name]:
                data_for_temp[metric_name] = metric_value
                data_for_temp[f"{metric_name}_called"] = params


    for temp, data in result_dic.items():
        def get_param(param_dict, key, default="N/A"):
            return param_dict.get(key, default) if param_dict else default
         
        print(f"{temp}\n") 
        emb_params = data['emb_called']
        print(f"Embedding: {data['emb']:.9f}")
        print(f"Emb threshold: {get_param(emb_params, 'emb threshold')} | iterations: {get_param(emb_params, 'amount iterations')}")
        print("-" * 10)

        print(f"Code BLEU: {data['bleu']:.9f}")
        bleu_params = data['bleu_called']
        print(f"Bleu threshhold: {get_param(bleu_params, 'bleu threshhold')} | iterations: {get_param(bleu_params, 'amount iterations')}")
        print("-" * 10)

        print(f"Test: {data['test']:.9f}")
        test_params = data['test_called']
        print(f"Iterations: {get_param(test_params, 'amount iterations')}")
        print("-" * 10)

        print(f"Combined: {data['comb']:.9f}")
        comb_params = data['comb_called']
        print(f"iterations: {get_param(comb_params, 'amount iterations')} | emb threshold: {get_param(comb_params, 'emb threshold')} | bleu threshhold: {get_param(comb_params, 'bleu threshhold')}")
        print("-" * 30)

if __name__ == "__main__":
    import re
    result_files = [#"results_multi_inference_deepseek-ai_deepseek-coder-6.7b-instruct_openai_humaneval.csv",
                    #"results_multi_inference_meta-llama_Meta-Llama-3.1-8B-Instruct_openai_humaneval.csv",
                    #"results_multi_inference_microsoft_Phi-4-mini-instruct_openai_humaneval.csv",
                    "results_multi_inference_Qwen_Qwen2.5-Coder-7B-Instruct_openai_humaneval.csv"
            ]   
    pattern = r"^results_multi_inference_[^_]+_(.*?)_openai_humaneval\.csv$"
    for file_name in result_files: 
        print(f"Model: {re.match(pattern, file_name).group(1)}")
        analayze_result_file(f"Results/{file_name}")
    
