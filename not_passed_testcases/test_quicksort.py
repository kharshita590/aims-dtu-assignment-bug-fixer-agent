import pytest
from python_testcases.load_testdata import load_json_testcases

if pytest.use_correct:
    from repaired_code.quicksort import quicksort
else:
    from python_programs.quicksort import quicksort


testdata = load_json_testcases(quicksort.__name__)

 
@pytest.mark.parametrize("input_data,expected", testdata)
def test_quicksort(input_data, expected):
    assert quicksort(*input_data) == expected
