"""Launcher for the stdin/stdout worker. The protocol lives in the package.

    python -u tools/worker.py            # this file
    python -u -m gacha_vision.worker     # same thing, if the package is installed

Kept as a path-launchable file because a caller in another language usually
has a path to a script rather than an installed package on its PYTHONPATH.
See `gacha_vision/worker.py` for the protocol.
"""

import sys
from pathlib import Path

# Python puts *this file's* directory on sys.path, not the caller's cwd, so
# without this the import below fails whenever the launcher runs from
# somewhere else -- which, for anything driving this, it always does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gacha_vision.worker import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
