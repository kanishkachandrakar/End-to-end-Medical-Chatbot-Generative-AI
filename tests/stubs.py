"""Shared test doubles.

Not a test module: named so pytest does not collect it, and declared as
supporting in tests/test_test_discovery.py so the check that every file in
here is collectable knows that is deliberate.
"""


class StubChain:
    """Stands in for the retrieval chain: records calls, answers as told.

    Three test modules each had their own copy, with three different
    signatures -- and one of those copies is where two tests accidentally
    ended up as methods and stopped running.
    """

    def __init__(self, answer: str = "an answer", raises: BaseException | None = None):
        self.answer = answer
        self.raises = raises
        self.calls: list = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.raises is not None:
            raise self.raises
        return {"answer": self.answer}
