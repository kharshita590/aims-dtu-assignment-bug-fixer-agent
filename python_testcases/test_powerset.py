import pytest
from python_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from repaired_code.powerset import powerset
else:
    from python_programs.powerset import powerset


testdata = load_json_testcases(powerset.__name__)

 
@pytest.mark.parametrize("input_data,expected", testdata)
def test_powerset(input_data, expected):
    assert powerset(*input_data) == expected
