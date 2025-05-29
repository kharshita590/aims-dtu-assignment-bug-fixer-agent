import pytest
from python_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from repaired_code.get_factors import get_factors
else:
    from python_programs.get_factors import get_factors


testdata = load_json_testcases(get_factors.__name__)

 
@pytest.mark.parametrize("input_data,expected", testdata)
def test_get_factors(input_data, expected):
    assert get_factors(*input_data) == expected
