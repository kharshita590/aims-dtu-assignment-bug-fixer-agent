import pytest
from python_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from repaired_code.gcd import gcd
else:
    from python_programs.gcd import gcd


testdata = load_json_testcases(gcd.__name__)

 
@pytest.mark.parametrize("input_data,expected", testdata)
def test_gcd(input_data, expected):
    assert gcd(*input_data) == expected
