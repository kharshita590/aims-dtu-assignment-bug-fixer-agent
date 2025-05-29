import pytest
from python_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from repaired_code.find_first_in_sorted import find_first_in_sorted
else:
    from python_programs.find_first_in_sorted import find_first_in_sorted


testdata = load_json_testcases(find_first_in_sorted.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_find_first_in_sorted(input_data, expected):
    assert find_first_in_sorted(*input_data) == expected
 