"""One-shot scaffold that created this project's initial file layout.

It has already been run. Running it again is safe: it only opens a path that is
missing or already zero-length, so nothing with content in it is ever lost.
(``src/__init__.py`` is legitimately empty, so that one would be re-created --
with the same empty contents.)

Kept as a record of how the project was laid out, not as a tool to use. Note
that it creates ``.env``: an empty one, which is harmless, but the real one is
gitignored and must not be committed. See SECURITY.md.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]

for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)
      logging.info(f"Creating directory; {filedir} for the file {filename}")

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass
         logging.info(f"Creating empty file: {filepath}")

   else:
      logging.info(f"{filename} is already created")
