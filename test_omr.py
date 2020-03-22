import os
from omr import get_answers


def test_recognizes_digitally_marked_answer_sheet():
    answers = get_answers(os.path.join("img", "rotated-answeredsheet.png"))
    assert answers[0] == ['A', 'C', 'C', 'E', 'N/A', 'N/A', 'A', 'N/A', 'N/A', 'N/A']


def test_recognizes_photo_of_digitally_marked_answer_sheet():
    answers = get_answers(os.path.join("img", "answered-sheet-photo.jpg"))
    assert answers[0] == ['A', 'C', 'C', 'E', 'N/A', 'N/A', 'A', 'N/A', 'N/A', 'N/A']

