"""Allow ``python -m transect_stitch`` to behave like the console script."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
