"""What the model is told before the question.

The only module here with no behaviour in it, and the one with the most
influence on what a visitor actually reads. Each clause is a decision:

  "three sentences maximum"     The reply goes into a chat bubble, and the
                                front end renders it as plain text with no
                                formatting, so a long answer arrives as a
                                wall.
  "write it as a definition"    The retrieved context is encyclopedia prose.
                                Asking for the same register keeps the answer
                                close to the passage it came from, which is
                                the only grounding this app has.
  "say that you don't know"     The alternative is a fluent answer with no
                                retrieved support, which is the failure an
                                empty index produces and the one /healthz
                                exists to surface.
  "only for medical terms"      Matches what the page says the bot is -- "looks
                                up medical terms in a reference encyclopedia"
                                -- and what the index contains. Nothing
                                enforces it but this sentence.

{context} is filled by create_stuff_documents_chain. It is the only
placeholder, and tests/test_prompt.py holds it to that: ChatPromptTemplate
treats any other brace as an unknown variable and fails the request rather
than the import.
"""

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the context does not contain the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Write it as a definition; do not include your own words or personal thoughts. "
    "Answer only for medical terms. If the question is not about a medical term, "
    "say that it is not a medical term and that you can't help with it."
    "\n\n"
    "{context}"
)
