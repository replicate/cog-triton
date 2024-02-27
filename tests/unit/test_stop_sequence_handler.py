import sys

sys.path.append(".")

from utils import StreamingTokenStopSequenceHandler


def test_process_with_no_stop_sequences():
    handler = StreamingTokenStopSequenceHandler(stop_sequences=[])
    output = handler("Hello")
    assert output == "Hello"


def test_token_stream_with_mismatched_stop_sequence():
    handler = StreamingTokenStopSequenceHandler(stop_sequences=["Candy"])
    output_tokens = ["H", "acker", "\n", "B", "ear", " +", " C", "andy", " ="]
    expected_output = ["H", "acker", "\n", "B", "ear", " +", " C", "andy", " ="]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)
        assert output == expected_token


def test_token_stream_with_matched_stop_sequence():

    handler = StreamingTokenStopSequenceHandler(stop_sequences=["\n"])
    output_tokens = ["H", "acker", "\n"]
    expected_output = ["H", "acker", None]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)

        assert output == expected_token

    output = handler.finalize()
    assert output == None


def test_token_stream_with_matched_two_part_stop_sequence():
    handler = StreamingTokenStopSequenceHandler(stop_sequences=[" Candy"])
    output_tokens = ["H", "acker", "\n", "B", "ear", " +", " C", "andy"]
    expected_output = ["H", "acker", "\n", "B", "ear", " +", None, None]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)
        assert output == expected_token

    output = handler.finalize()
    assert output == None


def test_token_stream_with_three_stop_sequences():

    handler = StreamingTokenStopSequenceHandler(stop_sequences=["hello", "world", "\n"])
    output_tokens = ["H", "acker", "\n"]
    expected_output = ["H", "acker", None]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)
        assert output == expected_token

    output = handler.finalize()
    assert output == None


def test_token_stream_with_overlapping_stop_sequence():

    handler = StreamingTokenStopSequenceHandler(stop_sequences=["Hello world", "\n"])
    output_tokens = ["Hello", " dog", ",", " how", "\n"]
    expected_output = [None, "Hello dog", ",", " how", None]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)
        assert output == expected_token

    output = handler.finalize()
    assert output == None


def test_token_stream_with_overlapping_stop_sequence():

    handler = StreamingTokenStopSequenceHandler(stop_sequences=[" not to be"])
    output_tokens = ["To", " be", " or", " not", " to"]

    expected_output = ["To", " be", " or", None, None, None]

    for token, expected_token in zip(output_tokens, expected_output):
        output = handler(token)
        assert output == expected_token

    output = handler.finalize()
    assert output == " not to"
