from Code.humanEval import HumanEval_connector
from Code.helpers import execute_generated_code

human_eval = HumanEval_connector("openai_humaneval")
first_line = human_eval.get_next_line()
generated_code = """
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    numbers.sort()
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] < threshold:
            return True
    return False
"""
generated_test_cases = '''
def generated_test_case(): 
    return 10
'''


class TestHumanEval:
    def test_human_eval_base_test_pass(self):
        assert execute_generated_code(generated_code, human_eval.get_test(first_line), generated_test_cases) == (1, 10)

    def test_human_eval_invalid_test(self):
        assert execute_generated_code(generated_code, human_eval.get_test(first_line), "") == (1, 404)

    def test_human_eval_base_test_fail(self):
        tests_false = """
        def check(candidate):
            assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
            assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
            assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
            assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
            assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
            assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == False
            assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
        """
        # Ensure that our function execution correctly returns the result of the provided tests
        # Ensure that, it returns -1, if the generated tests deviate from the generated structure.
        assert execute_generated_code(generated_code, tests_false, generated_test_cases) == (-1, 10)

    def test_human_eval_endless_loop_in_code(self):
        generated_code_inf = """
                from typing import List
                import time
                def has_close_elements(numbers: List[float], threshold: float) -> bool:
                    while True:
                        time.sleep(20);
                    return false
                """        # Ensure that test got interrupted after 5 seconds.
        # However still returns the correct result of the generated tests
        assert execute_generated_code(generated_code_inf, human_eval.get_test(first_line),
                                      generated_test_cases, 5) == (-1, 10)

    def test_human_eval_endless_loop_in_tests(self):
        generated_test_cases_inf = '''
        def generated_test_case(): 
            while True: 
                continue
        '''
        # Ensure that even with the generated tests time out, the method still returns
        assert execute_generated_code(generated_code, human_eval.get_test(first_line),
                                      generated_test_cases_inf, 5) == (1, -1)
