from multiprocessing import Process, Manager
import textwrap
import re
import logging

logger = logging.getLogger(__name__)


def extract_function_names(code):
    return re.findall(r"def (\w+)\(", code)


def execute_functions(run_function):
    try:
        return run_function()
    except AssertionError:
        return 0
    except Exception as e:
        output = "Another error:" + str(e) + "\n"
        print(output)
        return 0


def run_generated_testcase_code(code_generated, generated_testcase_code):
    try:
        tmp_namespace = {}
        exec(textwrap.dedent(code_generated), tmp_namespace)
        exec(textwrap.dedent(generated_testcase_code), tmp_namespace)
        test_func = tmp_namespace.get('generated_test_case')
        if not test_func:
            return 404
        return test_func()
    except Exception as e:
        print("Error running generated testcase code:", e)
        return -1


def run_dataset_test_code(code_generated, test_code):
    try:
        ns = {}
        exec(textwrap.dedent(code_generated), ns)
        exec(textwrap.dedent(test_code), ns)
        candidate_function = ns.get(extract_function_names(code_generated)[0])
        test_function = ns.get('check')

        def call_check_and_return_one():
            test_function(candidate_function)  # run test function
            return 1  # if no AssertionError, we consider all tests as passed

        return execute_functions(call_check_and_return_one)
    except Exception as e:
        print("Error running dataset test code:", e)
        return 0


def execute_code_mbpp_tests(code_generated, test_imports, tests):
    try:
        print(f"Code generated: \n{code_generated}\n")
        print(f"tests: \n{tests}\n")

        def run_test_list():
            # 1) Indent your string lines for readability...
            imports_str = """
                from math import *
                import math
            """

            # ...but dedent them so that Python code lines
            # start at the left margin when executed:
            imports_str = textwrap.dedent(imports_str).strip()

            # Combine the test imports similarly
            tests_import_str = "\n".join(test_imports) if test_imports else ""

            # Turn the list of assert statements into one string
            tests_str = "\n".join(tests)

            # Dedent code_generated as well
            code_generated_dedented = textwrap.dedent(code_generated).strip()

            # 2) Build the final code block in one triple-quoted string
            combined_code = f"""
{imports_str}

{tests_import_str}

{code_generated_dedented}

{tests_str}
"""
            # 3) Dedent the final combined string again (if desired),
            #    and then exec:
            final_code = textwrap.dedent(combined_code).strip()
            exec(final_code, {})

            return 1  # return 1 on successful execution

        return execute_functions(run_test_list)

    except Exception as e:
        print("Error running MBPP tests:", e)
        return 0


def target_func(func, args, kwargs, result):
    try:
        result["value"] = func(*args, **kwargs)
    except Exception as e:
        print("Error in target_func:", e)
        result["value"] = 0


# Uses process to execute LLM generated code and tests in a separate process.
# A lot slower than just executing the code. However, ensures that the code gets terminated even if the LLM generates
# an endless loop.
def run_timed(func, args=(), timout_return=0, timeout=3):
    """
    Runs a function in a separate process with a timeout and returns its result.
    """
    manager = Manager()
    result = manager.dict()
    process = Process(
        target=target_func,
        args=(func, args, {}, result)
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        #print("The process is still running after the timeout!")
        process.terminate()
        process.join()
        return timout_return
    return result.get("value", 0)


def execute_generated_code_normal(code_generated, test_code, timeout=5):
    output_dataset_tests = run_timed(
        run_dataset_test_code,
        args=(code_generated, test_code),
        timeout=timeout
    )
    if output_dataset_tests is None or output_dataset_tests == -3:
        output_dataset_tests = 0

    return output_dataset_tests


def execute_generated_tests(code_generated, generated_tests, timeout=5):
    output_generated_tests = run_timed(
        run_generated_testcase_code,
        args=(code_generated, generated_tests),
        timout_return=-1,
        timeout=timeout
    )
    if output_generated_tests is None or output_generated_tests == -3:
        output_generated_tests = -1

    return output_generated_tests


def execute_generated_code_mbpp(code_generated, test_imports, tests, timeout=5):
    output_mbpp_tests = run_timed(
        execute_code_mbpp_tests,
        args=(code_generated, test_imports, tests),
        timeout=timeout
    )
    if output_mbpp_tests is None or output_mbpp_tests == -3:
        output_mbpp_tests = 0

    return output_mbpp_tests


def get_api_key():
    with open('API.txt', 'r') as file:
        return file.read()


def get_api_key_voyage():
    with open('VoyageAPI', 'r') as file:
        return file.read()
