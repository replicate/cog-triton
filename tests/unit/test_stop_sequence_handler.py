import sys

sys.path.append(".")

from utils.utils import StreamingTextStopSequenceHandler


def test_process_with_no_stop_sequences():
    handler = StreamingTextStopSequenceHandler(stop_sequences=[])
    output = handler("Hello")
    assert output == "Hello"


def test_process_with_single_stop_sequence():
    handler = StreamingTextStopSequenceHandler(stop_sequences=["world"])
    output = handler("Hello")
    assert output == "Hello"

    output = handler(" ")
    assert output == " "

    result = handler("world")
    assert result == None


def test_with_multiple_stop_sequences():
    handler = StreamingTextStopSequenceHandler(stop_sequences=["world", "universe"])
    output = handler("Hello")
    assert output == "Hello"

    output = handler(" ")
    assert output == " "

    output = handler("universe")
    assert output == None


def test_with_partial_stop_sequence():
    handler = StreamingTextStopSequenceHandler(stop_sequences=["dogs of the world"])
    output = handler("Hello")
    assert output == "Hello"

    output = handler(" ")
    assert output == " "

    output = handler("dogs")
    assert output == None

    output = handler(" of the universe")
    assert output == "dogs of the universe"


def test_with_overlapping_stop_sequences():
    handler = StreamingTextStopSequenceHandler(
        stop_sequences=["dogs  and cats", "dogs and cats"]
    )
    output = handler("Hello")
    assert output == "Hello"

    output = handler(" ")
    assert output == " "

    output = handler("dogs")
    assert output == None

    output = handler(" ")
    assert output == None

    output = handler("and cats")
    assert output == None


def test_final_cache_clearing():
    handler = StreamingTextStopSequenceHandler(stop_sequences=["dogs world"])
    output = handler("Hello")
    assert output == "Hello"

    output = handler("dogs")
    assert output == None

    output = handler(" world, how are you?")
    assert output == None

    output = handler.finalize()
    assert output == None

    output = handler("Hello ")
    assert output == "Hello "

    output = handler("dog ")
    assert output == "dog "

    output = handler("world")
    assert output == "world"


def test_wtf():
    handler = StreamingTextStopSequenceHandler(stop_sequences=["stop here"])
    output = handler("Hello")
    assert output == "Hello"

    output = handler(" this stop here blue")
    assert output == None

    output = handler.finalize()
    assert output == " this "
