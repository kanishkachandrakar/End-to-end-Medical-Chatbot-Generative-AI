from setuptools import find_packages, setup

setup(
    name="generative-ai-project",
    version="0.1.0",
    description="Retrieval-augmented medical question-answering chatbot",
    author="Kanishka Chandrakar",
    author_email="kanishka.chandrakar02@gmail.com",
    license="MIT",
    # Constrained to src: a bare find_packages() also picks up tests/,
    # because it has an __init__.py so the suite can `from tests.stubs import
    # ...`. top_level.txt then listed "src" and "tests", so `pip install -e .`
    # put a package called tests into the environment -- ready to collide with
    # any other project's, and ready to ship in a wheel.
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[],
)
