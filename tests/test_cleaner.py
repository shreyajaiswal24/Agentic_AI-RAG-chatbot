from app.ingestion.cleaner import clean_text


def test_removes_running_header_and_page_numbers():
    raw = "AGENTIC AI FOR EXECUTIVES\nReal content here.\n12\nMore content."
    assert clean_text(raw) == "Real content here.\n\nMore content."


def test_replaces_broken_bullets_and_smart_quotes():
    raw = "� First point\n� Second ’quoted’"
    assert clean_text(raw) == "• First point\n• Second 'quoted'"


def test_collapses_whitespace_but_keeps_paragraphs():
    raw = "Para one   with   gaps.  \n\n\n\nPara two."
    assert clean_text(raw) == "Para one with gaps.\n\nPara two."


def test_does_not_strip_numbers_inside_sentences():
    raw = "Productivity rose 30% in 2024."
    assert clean_text(raw) == raw
