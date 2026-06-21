"""Sentence segmentation drives TTS latency: flush as early as possible, never mid-number."""
from main import Daemon

split = Daemon._sentence_end


def test_flushes_on_terminal_period():
    assert split("Hello there.") == len("Hello there.") - 1


def test_flushes_on_period_before_space():
    s = "First sentence. Second"
    assert split(s) == s.index(".")


def test_no_boundary_returns_minus_one():
    assert split("no punctuation here") == -1


def test_decimal_numbers_are_not_boundaries():
    assert split("pi is 3.14159 and counting") == -1


def test_question_and_exclamation():
    assert split("Really? Yes") == "Really?".index("?")
    assert split("Go! Now") == "Go!".index("!")


def test_newline_boundary_requires_trailing_whitespace():
    assert split("line one\nline two") == -1
    s = "line one\n\nline two"
    assert split(s) == s.index("\n")


def test_comma_flush_only_for_long_chunks():
    short = "a, b"
    assert split(short) == -1
    long = "x" * 40 + ", " + "y" * 40
    assert split(long) == long.index(",")
