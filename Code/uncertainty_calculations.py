# File that contains all the needed methods for our uncertainty calculations
from enum import Enum
import math
import black
from codebleu import calc_codebleu
from embedding import calculate_similarity_template


class Similarity_Calculations(Enum):
    SYNTACTIC = 1
    EMBEDDING = 2
    TEST_BASED = 3
    COMBINED = 4


"""
Method that returns a list of different code clusters for an initial list of different code samples.
It clusters the methods according to similarity and preserves transitivity.
The code lists consists of pairs of code snippets together with the entropy values for the snippet
An entry in the code list consists of: 
(the generated code, summed up token probabilities, 
"""


def cluster(code_list, BLEU_THRESHOLD, EMB_THRESHOLD, similarity_method):
    clusters = []
    for current_code_snippet in code_list:
        cluster_to_add = []
        # Iterate through existing clusters to check similarity
        for current_cluster_index, cl in enumerate(clusters):
            for (cluster_code_snippet, _, _, test_passed_snipped, emb_snipped) in cl:
                if is_similar(cluster_code_snippet, current_code_snippet[0],
                              test_passed_snipped, current_code_snippet[3], 
                              emb_snipped, current_code_snippet[4], BLEU_THRESHOLD,
                              EMB_THRESHOLD, similarity_method):
                    if current_cluster_index not in cluster_to_add:
                        cluster_to_add.append(current_cluster_index)
                    break
        if len(cluster_to_add) == 1:
            clusters[cluster_to_add[0]].append(current_code_snippet)
        elif len(cluster_to_add) >= 2:
            clusters = merge_clusters(cluster_to_add, clusters)
            clusters[cluster_to_add[0]].append(current_code_snippet)
        else:
            clusters.append([current_code_snippet])
    return clusters


def merge_clusters(index_list, clusters):
    merged_list = []
    result = []
    for index, cl in enumerate(clusters):
        if index in index_list:
            merged_list += clusters[index]
        else:
            result.append(clusters[index])
    if len(merged_list) > 0:
        result.append(merged_list)
    return result


# method that returns 0 if code is similar and 1 if is not similar and calculates similarity based on the selected mode

def is_similar(code_sample_1, code_sample_2,
               tests_passed_sample_1, tests_passed_sample_2,
               embedding_vector_snipped1, embedding_vector_snipped2,
               BLEU_THRESHOLD, EMB_THRESHOLD,
               embedding_mode):
    if embedding_mode == Similarity_Calculations.SYNTACTIC:
        return calc_codebleu([code_sample_1],
                             [code_sample_2],
                             lang="python",
                             weights=(0.00, 0.00, 1, 0),
                             tokenizer=None)['codebleu'] > BLEU_THRESHOLD
    elif embedding_mode == Similarity_Calculations.EMBEDDING:
        return calculate_similarity_template(embedding_vector_snipped1, embedding_vector_snipped2) > EMB_THRESHOLD
    elif embedding_mode == Similarity_Calculations.TEST_BASED:
        return tests_passed_sample_1 == tests_passed_sample_2
    else:
        if (tests_passed_sample_1 == -1 and tests_passed_sample_2 == -1) or (tests_passed_sample_1 == 404 and tests_passed_sample_2 == 404):
            return calculate_similarity_template(embedding_vector_snipped1, embedding_vector_snipped2) > EMB_THRESHOLD
        return tests_passed_sample_1 == tests_passed_sample_2


# Returns the product of the token probs. for a single code snipped, resulting in sequence probabilities. (equation 6)
def calculate_snipped_probabilities_and_length(code_snipped, probs, result_passed_tests, embeddings):
    return (code_snipped, math.prod(value[0] for value in probs), len(probs),
            result_passed_tests, embeddings)


# Returns the product of the snipped probabilities. (equation 7)
def calculate_probabilities_equiv_classes(equiv_class):
    return sum(snipped[1] for snipped in equiv_class)


# Similar to tbe previous function.
# However also applies length normalization to the calculation of the snipped probabilities.
def calculate_probabilities_equiv_classes_length_normalized(equiv_class):
    prob_sum_normalized = 0.0
    for _, prop, snipped_length, _, _ in equiv_class:
        if snipped_length > 0:
            prob_sum_normalized += prop / snipped_length
        else:
            raise Exception("Invalid length when calculating equivalence classes")
    return prob_sum_normalized


# Calculations Equiv. Entropy and also applies monte carlo integration. (equation 8)
def calculate_entropy_for_equiv_classes(equiv_classes, include_length_normalization=False):
    sum_log_probs = 0
    n_equiv_classes = len(equiv_classes)
    for ec in equiv_classes:
        if include_length_normalization:
            sum_log_probs += math.log(calculate_probabilities_equiv_classes_length_normalized(ec))
        else:
            sum_log_probs += math.log(calculate_probabilities_equiv_classes(ec))
    return -sum_log_probs / n_equiv_classes


# The following methods are used for single inference calculations

# Calculates the avg token prob for an output (equation 1)
def calculate_avg_prob_single_inference(probabilities):
    avg_log_prob = 0
    for prob, token in probabilities:
        if prob > 0:
            avg_log_prob += math.log(prob)
        else:
            raise ValueError("cant take log of value <= 0")

    return -(avg_log_prob / len(probabilities))


# Calculates the max token prob for an output (equation 2)

def calculate_max_prob_single_inference(probabilities):
    max_log_prob = 0
    for prob, token in probabilities:
        if prob > 0:
            neg_log_prob = -math.log(prob)
        else:
            raise ValueError("cant take log of value <= 0")
        if neg_log_prob > max_log_prob:
            max_log_prob = neg_log_prob

    return max_log_prob


# calculates the entropy for single token position
def calculate_entropy_single_token_position(token_iteration):
    return -sum(p * math.log2(p) for p, token in token_iteration if p > 0)


# calculates the avg entropy for a single inference example. (equation 3)
def calculate_avg_entropy_single_inference(top_k_tokens):
    return sum(calculate_entropy_single_token_position(prob_list) for prob_list in top_k_tokens) / len(top_k_tokens)


# calculates the max entropy for a single inference example. (equation 4)

def calculate_max_entropy_single_inference(top_k_tokens):
    max_entropy_value = 0
    for prob_list in top_k_tokens:
        current_entropy_value = calculate_entropy_single_token_position(prob_list)
        if current_entropy_value > max_entropy_value:
            max_entropy_value = current_entropy_value

    return max_entropy_value


if __name__ == '__main__':
    print(Similarity_Calculations.SYNTACTIC)
