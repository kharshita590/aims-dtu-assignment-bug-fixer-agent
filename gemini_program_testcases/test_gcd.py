import pytest
from gemini_program_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from gemini_repaired_programs.gcd import gcd
else:
    from python_programs.gcd import gcd


testdata = load_json_testcases(gcd.__name__)

 
@pytest.mark.parametrize("input_data,expected", testdata)
def test_gcd(input_data, expected):
    assert gcd(*input_data) == expected
