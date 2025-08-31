from Code.uncertainty_calculations import *
import pytest


# Test class to test uncertainty calculations
class Test_UE_calcs:

    def test_merge_standard(self):
        assert ([[5], [6], [1, 2, 4, 3, 8]] == merge_clusters([0, 1, 2], [[1, 2], [4, 3], [8], [5], [6]]))
        assert ([[8], [5], [6], [1, 2, 4, 3]] == merge_clusters([0, 1], [[1, 2], [4, 3], [8], [5], [6]]))
        assert ([[1, 2], [4, 3], [8, 5, 6]] == merge_clusters([2, 3, 4], [[1, 2], [4, 3], [8], [5], [6]]))
        assert ([[1, 2, 3, 4, 5]] == merge_clusters([0, 1], [[1, 2, 3, 4], [5]]))

    def test_merge_edge(self):
        assert ([[6, 5, 4, 3, 2, 1, 0]] == merge_clusters([0, 1, 2, 3, 4, 5, 6], [[6], [5], [4], [3], [2], [1], [0]]))
        assert ([[6], [5], [4], [3], [2], [1], [0]] == merge_clusters([], [[6], [5], [4], [3], [2], [1], [0]]))
        assert ([[1, 2, 3, 4, 5]] == merge_clusters([0], [[1, 2, 3, 4, 5]]))
        assert ([[5], [6], [1, 2, 4, 3, 8]] == merge_clusters([0, 1, 1, 2, 2, 0], [[1, 2], [4, 3], [8], [5], [6]]))

    def test_cluster_with_mocked_similarity(self, mocker):
        sample_embedding = (1, 2)
        code_list = [("A", 0.1, 1, 10, sample_embedding), ("B", 0.1, 1, 10, sample_embedding),
                     ("C", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)]

        def pairwise_similarity_basic(code_sample_1, code_sample_2, _tests_passed_sample_1, _tests_passed_sample_2,
                                      _embedding_vector_snipped1, _embedding_vector_snipped2, _BLEU_THRESHOLD,
                                      _EMB_THRESHOLD, embedding_mode):
            similarity_map = {
                "A": ["B"],
                "C": ["D"],
            }
            return code_sample_2 in similarity_map.get(code_sample_1, []) or code_sample_1 in similarity_map.get(
                code_sample_2, [])

        def always_dissimilar(*args):
            return False

        def single_pair_similarity(code_sample_1, code_sample_2, _tests_passed_sample_1, _tests_passed_sample_2,
                                   _embedding_vector_snipped1, _embedding_vector_snipped2, _BLEU_THRESHOLD,
                                   _EMB_THRESHOLD,
                                   _embedding_mode):
            similarity_map = {
                "A": ["D"],
            }
            return code_sample_2 in similarity_map.get(code_sample_1, []) or code_sample_1 in similarity_map.get(
                code_sample_2, [])

        def always_similar(*args):
            return True

        def expanded_similarity(code_sample_1, code_sample_2, _tests_passed_sample_1, _tests_passed_sample_2,
                                _embedding_vector_snipped1, _embedding_vector_snipped2, _BLEU_THRESHOLD, _EMB_THRESHOLD,
                                _embedding_mode):
            similarity_map = {
                "A": ["B"],
                "B": ["C", "D"],
            }
            return code_sample_2 in similarity_map.get(code_sample_1, []) or code_sample_1 in similarity_map.get(
                code_sample_2, [])

        def group_similarity(code_sample_1, code_sample_2, _tests_passed_sample_1, _tests_passed_sample_2,
                             _embedding_vector_snipped1, _embedding_vector_snipped2, _BLEU_THRESHOLD, _EMB_THRESHOLD,
                             _embedding_mode):
            similarity_map = {
                "A": ["C"],
                "B": ["D"],
                "C": ["D"],
            }
            return code_sample_2 in similarity_map.get(code_sample_1, []) or code_sample_1 in similarity_map.get(
                code_sample_2, [])

        def broader_group_similarity(code_sample_1, code_sample_2, _tests_passed_sample_1, _tests_passed_sample_2,
                                     _embedding_vector_snipped1, _embedding_vector_snipped2, _BLEU_THRESHOLD,
                                     _EMB_THRESHOLD, _embedding_mode):
            similarity_map = {
                "A": ["C", "D"],
            }
            return code_sample_2 in similarity_map.get(code_sample_1, []) or code_sample_1 in similarity_map.get(
                code_sample_2, [])

        expected_result_list = [
            [
                [("A", 0.1, 1, 10, sample_embedding), ("B", 0.1, 1, 10, sample_embedding)],
                [("C", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding)], [("B", 0.1, 1, 10, sample_embedding)],
                [("C", 0.1, 1, 10, sample_embedding)], [("D", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)],
                [("B", 0.1, 1, 10, sample_embedding)], [("C", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding), ("B", 0.1, 1, 10, sample_embedding),
                 ("C", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding), ("B", 0.1, 1, 10, sample_embedding),
                 ("C", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding), ("C", 0.1, 1, 10, sample_embedding),
                 ("B", 0.1, 1, 10, sample_embedding), ("D", 0.1, 1, 10, sample_embedding)]
            ],
            [
                [("A", 0.1, 1, 10, sample_embedding), ("C", 0.1, 1, 10, sample_embedding),
                 ("D", 0.1, 1, 10, sample_embedding)], [("B", 0.1, 1, 10, sample_embedding)]
            ],
        ]
        similarity_functions = [pairwise_similarity_basic, always_dissimilar, single_pair_similarity, always_similar,
                                expanded_similarity, group_similarity, broader_group_similarity]
        for index, expected_result in enumerate(expected_result_list):
            mocker.patch("Code.uncertainty_calculations.is_similar", side_effect=similarity_functions[index])

            result = cluster(code_list, 0, 0)
            assert result == expected_result

    def test_calculate_snipped_probabilities_and_length(self):
        code = ""
        sample_embedding = (1, 2, 3, 4)
        token_probs = [(0.75, 'def'), (0.5, ' add'), (1.0, '(a'), (0.25, ',')]
        assert ((code, 0.09375, 4, 10, sample_embedding) ==
                calculate_snipped_probabilities_and_length(code, token_probs, 10, sample_embedding))

        token_probs = [(0.02, 'def'), (0.75, ' add'), (0.99, '(a'), (0.55, ','), (0.55, ','), (0.55, ',')]
        assert ((code, 0.0024706687500000006, 6, 10, sample_embedding) ==
                calculate_snipped_probabilities_and_length(code, token_probs, 10, sample_embedding))

        token_probs = [(0.75, 'def'), (0, ' add'), (1.0, '(a'), (0.25, ',')]
        assert ((code, 0, 4, 10, sample_embedding) ==
                calculate_snipped_probabilities_and_length(code, token_probs, 10, sample_embedding))

        token_probs = [(-1, 'def'), (1, ' add'), (1.0, '(a'), (0.25, ',')]
        assert ((code, -0.25, 4, 10, sample_embedding) ==
                calculate_snipped_probabilities_and_length(code, token_probs, 10, sample_embedding))

    def test_calculate_probabilities_equiv_classes(self):
        sample_embedding = (1, 2)
        equiv_class_v1 = [("A", 0.5, 10, sample_embedding), ("B", 0.5, 10, sample_embedding),
                          ("C", 0.5, 10, sample_embedding)]
        equiv_class_v2 = [("A", 0.5, 10, sample_embedding), ("B", -0.5, 10, sample_embedding),
                          ("C", 1, 10, sample_embedding)]
        equiv_class_v3 = [("A", 2, 10, sample_embedding), ("B", -2, 10, sample_embedding),
                          ("C", 0, 10, sample_embedding), ("D", 0, 10, sample_embedding)]
        equiv_class_v4 = [("A", 0, 10, sample_embedding), ("B", 0, 10, sample_embedding),
                          ("C", 0, 10, sample_embedding), ("C", 0, 10, sample_embedding)]
        equiv_class_v5 = [["A", -1, 10, sample_embedding], ["B", -1, 10, sample_embedding],
                          ["C", -1, 10, sample_embedding], ["C", -1, 10, sample_embedding]]

        assert calculate_probabilities_equiv_classes(equiv_class_v1) == 1.5
        assert calculate_probabilities_equiv_classes(equiv_class_v2) == 1
        assert calculate_probabilities_equiv_classes(equiv_class_v3) == 0
        assert calculate_probabilities_equiv_classes(equiv_class_v4) == 0
        assert calculate_probabilities_equiv_classes(equiv_class_v5) == -4

    def test_calculate_probabilities_equiv_classes_length_normalized(self):
        sample_embedding = (1, 2, 3, 4)
        equiv_class_v1 = [("A", 10, 10, sample_embedding), ("B", 10, 10, sample_embedding),
                          ("C", 10, 10, sample_embedding)]
        equiv_class_v2 = [("A", 3, 2, sample_embedding), ("B", 4, 2, sample_embedding),
                          ("C", 1, 1, sample_embedding)]
        equiv_class_v3 = [("A", 2, 2, sample_embedding), ("B", 4, 4, sample_embedding),
                          ("C", -3, 3, sample_embedding), ("D", -5, 5, sample_embedding)]
        equiv_class_v4 = [("A", 0, 10, sample_embedding), ("B", 0, 10, sample_embedding),
                          ("C", 0, 10, sample_embedding), ("C", 0, 10, sample_embedding)]
        equiv_class_v5 = [["A", -1, 10, sample_embedding], ["B", -1, 10, sample_embedding],
                          ["C", -1, 0, sample_embedding], ["C", -1, 10, sample_embedding]]

        assert 3 == calculate_probabilities_equiv_classes_length_normalized(equiv_class_v1)
        assert 4.5 == calculate_probabilities_equiv_classes_length_normalized(equiv_class_v2)
        assert 0 == calculate_probabilities_equiv_classes_length_normalized(equiv_class_v3)
        assert 0 == calculate_probabilities_equiv_classes_length_normalized(equiv_class_v4)
        with pytest.raises(Exception):
            calculate_probabilities_equiv_classes_length_normalized(equiv_class_v5)

    def test_calculate_semantic_entropy(self):
        sample_embedding = ()
        clusters_1 = [[("A", 0.25, 10, sample_embedding), ("B", 0.25, 10, sample_embedding),
                       ("C", 0.5, 10, sample_embedding)]]
        clusters_2 = [[("A", 5, 10, sample_embedding), ("B", 5, 10, sample_embedding)],
                      [("C", 10, 10, sample_embedding)]]
        clusters_3 = [[("A", 10, 10, sample_embedding), ("B", 0, 10, sample_embedding)],
                      [("C", 1, 2, sample_embedding)]]
        assert 0 == calculate_entropy_for_equiv_classes(clusters_1)
        assert 0 == calculate_entropy_for_equiv_classes(clusters_2, True)
        assert 0.346574 == round(calculate_entropy_for_equiv_classes(clusters_3, True), 6)

    def test_calculate_avg_and_max_prob_single_inference(self):
        token_prob_list_1 = [(0.5, "token1"), (0.2, "token2"), (0.6, "token3"), (0.2, "token4"), (0.1, "token5")]
        token_prob_list_2 = [(0, "token1"), (0, "token2"), (0, "token3"), (0, "token4"), (0, "token5")]
        token_prob_list_3 = [(0.3, "token1"), (0.5, "token2"), (0.1, "token3"), (0.4, "token4"), (0.5, "token5")]
        token_prob_list_4 = [(1, "token1"), (1, "token2"), (1, "token3"), (1, "token4"), (1, "token5")]

        assert 1.34509 == round(calculate_avg_prob_single_inference(token_prob_list_1), 5)
        with pytest.raises(ValueError, match="cant take log of value <= 0"):
            calculate_avg_prob_single_inference(token_prob_list_2)
        assert 1.16183 == round(calculate_avg_prob_single_inference(token_prob_list_3), 5)
        assert 0 == round(calculate_avg_prob_single_inference(token_prob_list_4), 5)

        # tests for max log prop
        assert 2.30259 == round(calculate_max_prob_single_inference(token_prob_list_1), 5)
        with pytest.raises(ValueError, match="cant take log of value <= 0"):
            calculate_max_prob_single_inference(token_prob_list_2)
        assert 2.30259 == round(calculate_max_prob_single_inference(token_prob_list_3), 5)
        assert 0 == round(calculate_max_prob_single_inference(token_prob_list_4), 5)

    def test_entropy_single_token_position(self):
        token_prop_list_1 = [(0.5, "def "), (0.5, "add:"), (0.5, "a +"), (0.5, "b")]
        token_prop_list_2 = [(0.5, "def "), (1, "add:"), (1, "a +"), (0.5, "b")]
        token_prop_list_3 = [(1, "def "), (1, "add:"), (1, "a -"), (1, "b")]
        token_prop_list_4 = [(0.5, "def "), (1, "add:"), (1, "a -"), (1, "b")]

        assert 2 == calculate_entropy_single_token_position(token_prop_list_1)
        assert 1 == calculate_entropy_single_token_position(token_prop_list_2)
        assert 0 == calculate_entropy_single_token_position(token_prop_list_3)
        assert 0.5 == calculate_entropy_single_token_position(token_prop_list_4)

    def test_calculate_avg_and_max_entropy_single_inference(self):
        top_k_tokens_1 = [(0.5, "def "), (0.5, "add:"), (0.5, "a +"), (0.5, "b")]
        top_k_tokens_2 = [(0.5, "def "), (1, "add:"), (1, "a +"), (0.5, "b")]
        top_k_tokens_3 = [(1, "def "), (1, "add:"), (1, "a -"), (1, "b")]
        top_k_tokens_4 = [(0.5, "def "), (1, "add:"), (1, "a -"), (1, "b")]

        llm_output = [top_k_tokens_1, top_k_tokens_2, top_k_tokens_3, top_k_tokens_4]

        assert 2 == calculate_max_entropy_single_inference(llm_output)
        assert 0.875 == calculate_avg_entropy_single_inference(llm_output)
