"""A long-lived reader that another process talks to over stdin/stdout.

Spawning a fresh interpreter per spawn costs about 0.8s of startup before any
work happens, which is most of the budget for reacting to a live drop. This
stays up instead: one process, one import, ~23ms per answer.

Protocol -- newline-delimited JSON in both directions, one request per line:

    ->  {"image": "<base64>", "expected": 2}
    <-  {"slots": [2]}

`expected` is optional; omit it to let segmentation decide the card count.
Every request gets exactly one response line, so a caller can pipeline. A
request that cannot be read answers with an error rather than an empty
result, because "nothing worth claiming" and "the image was broken" must
never look the same:

    <-  {"error": "could not decode image from bytes"}

Anything unexpected on stdin is answered, never fatal: the worker only exits
when stdin closes. Run it with `python -u` so replies are not buffered.

    python -u tools/worker.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# Python puts *this file's* directory on sys.path, not the caller's cwd, so
# without this the import below fails whenever the launcher runs from
# somewhere else -- which, for anything driving this, it always does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    # Imported here, not at module scope, so the ready line below is only
    # printed once the model and OpenCV are actually loaded -- a caller that
    # waits for it knows the next request will be fast.
    from gacha_vision import pick

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            raw = base64.b64decode(req["image"], validate=True)
            slots = pick(raw, expected=req.get("expected"))
            reply = {"slots": slots}
        except Exception as exc:                      # never die on one bad request
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
