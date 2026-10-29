"""Everything a visitor can be shown.

Collected here because it is copy, not logic: someone rewording the refusals
should not have to read two modules and a JavaScript file to find them all, and
whoever reviews the wording should be able to see it as a set.

They are deliberately free of internal vocabulary. "The retrieval chain could
not be built" is the truth and means nothing to the person reading it; what
reaches the page says what happened and what to do instead. The operator-facing
version of the same state lives in /healthz and the log, where naming the
component is the point.
"""

# Asked for something, got nothing usable.
NO_QUESTION = "Please type a question."
NO_ANSWER = "I don't have an answer for that."

# Asked for too much.
TOO_LONG = "That question is too long -- please keep it under {limit} characters."
# {wait} is rendered by src.text.seconds_in_words. The limiter knows exactly
# how far away the next token is -- the Retry-After header beside this message
# has carried that number all along -- and "shortly" was the vaguest available
# way to say it.
TOO_BUSY = (
    "This demo answers a limited number of questions a minute -- "
    "please try again in {wait}."
)

# Something on this side is wrong.
RATE_LIMITED = "I am being rate limited right now -- please try again in a moment."
TIMED_OUT = "That took too long to answer -- please try again."
UNAVAILABLE = "Sorry, I could not answer that right now. Please try again."

# Every string above, for the test that holds them to being plain.
ALL = (
    NO_QUESTION,
    NO_ANSWER,
    TOO_LONG,
    TOO_BUSY,
    RATE_LIMITED,
    TIMED_OUT,
    UNAVAILABLE,
)
