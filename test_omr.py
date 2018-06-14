import os
from omr import get_answers


def test_omr():
    answers = get_answers(os.path.join("img", "answered-sheet-photo.jpg"))
    assert answers[0] == ['A', 'C', 'C', 'E', 'N/A', 'N/A', 'A', 'N/A', 'N/A', 'N/A']
