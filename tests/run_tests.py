"""Run the smoke tests without letting Qt tear the interpreter down.

Every test can pass and the process can still die:

    Ran 6 tests in 46.160s
    OK
    Segmentation fault (core dumped)

That crash happens after the results are in, while Python is dismantling
modules and the remaining Qt objects are finalised in an order nobody
controls. It says nothing about the code under test, but it does set a
non-zero exit status and turn the job red.

So: run the suite, report it, flush, and leave via os._exit, which skips
interpreter shutdown entirely. The exit status still reflects real results --
a genuine failure or a crash during the run is still a failure.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sys.path.insert(0, ROOT)
    suite = unittest.TestLoader().discover(
        start_dir=os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
