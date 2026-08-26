"""``python -m gacha_vision`` entry point.

The __main__ guard is load-bearing: batch runs use the "spawn" start method,
which re-imports this module in every worker, and without it each worker
would re-run the CLI.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
