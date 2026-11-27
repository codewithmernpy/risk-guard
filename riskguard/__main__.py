"""Module entry point for `python -m riskguard`."""

from __future__ import annotations

import sys

from riskguard.cli import main

if __name__ == "__main__":
    sys.exit(main())
