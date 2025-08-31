from Code.mbpp import Mbpp_connector
from Code.helpers import execute_generated_code_mbpp

generated_code = """
          def remove_Occ(s, ch):
              # Remove the first occurrence of the character
              for i in range(len(s)):
                  if s[i] == ch:
                      s = s[:i] + s[i + 1:]
                      break

              # Remove the last occurrence of the character
              for i in range(len(s) - 1, -1, -1):
                  if s[i] == ch:
                      s = s[:i] + s[i + 1:]
                      break
              return s

          """
mbpp = Mbpp_connector("mbpp")
first_line = mbpp.get_next_line()
generated_test_cases = '''
               def generated_test_case(): 
                   return 10
               '''


class TestMBPP:
    def test_human_eval_base_test_pass(self):
        assert mbpp.get_test_import(first_line) == []
        assert execute_generated_code_mbpp(generated_code, mbpp.get_test_import(first_line),
                                           mbpp.get_test(first_line), generated_test_cases) == (1, 10)

    def test_human_eval_invalid_test(self):
        assert mbpp.get_test_import(first_line) == []
        assert execute_generated_code_mbpp(generated_code, mbpp.get_test_import(first_line),
                                           mbpp.get_test(first_line), "") == (1, 404)

    def test_human_eval_base_test_fail(self):
        tests_false = ['assert remove_Occ("hello","l") == "heo"',
                       'assert remove_Occ("abcda","a") == "__CHANGED__VALUE"',
                       'assert remove_Occ("PHP","P") == "H"']
        # Ensure that our function execution correctly returns the result of the provided tests
        # Ensure that, it returns -1, if the generated tests deviate from the generated structure.
        assert execute_generated_code_mbpp(generated_code, mbpp.get_test_import(first_line),
                                           tests_false, "") == (-1, 404)

    def test_human_eval_endless_loop_in_code(self):
        generated_code_inf_loop = """
                def remove_Occ(s, ch):
                    import time
                    while True:
                        time.sleep(20);
                    return false
                """
        # Ensure that test got interrupted after 5 seconds.
        # However still returns the correct result of the generated tests
        assert execute_generated_code_mbpp(generated_code_inf_loop, mbpp.get_test_import(first_line),
                                           mbpp.get_test(first_line), generated_test_cases) == (-1, 10)

    def test_human_eval_endless_loop_in_tests(self):
        generated_test_cases_inf = '''
        def generated_test_case(): 
            while True: 
                continue
        '''
        # Ensure that even with the generated tests time out, the method still returns
        assert execute_generated_code_mbpp(generated_code, mbpp.get_test_import(first_line),
                                           mbpp.get_test(first_line), generated_test_cases_inf) == (1, -1)
